import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from speech_to_text import transcribe_file_with_whisper


class NAOContinuousAudioSource:
    """Stream raw PCM audio continuously from NAO over SSH."""

    def __init__(
        self,
        *,
        nao_ip: str,
        nao_user: str = "nao",
        samplerate: int = 16000,
        frame_ms: int = 30,
        channels: int = 1,
        queue_size: int = 512,
        arecord_device: str = "",
    ):
        self.nao_ip = nao_ip
        self.nao_user = nao_user
        self.samplerate = int(samplerate)
        self.frame_ms = int(frame_ms)
        self.channels = int(channels)
        self.arecord_device = (arecord_device or "").strip()

        if self.samplerate <= 0:
            raise RuntimeError(f"Invalid NAO samplerate: {self.samplerate}")
        if self.frame_ms <= 0:
            raise RuntimeError(f"Invalid NAO frame_ms: {self.frame_ms}")
        if self.channels != 1:
            raise RuntimeError("NAOContinuousAudioSource currently supports mono only")

        self.frame_samples = max(1, int(self.samplerate * self.frame_ms / 1000))
        self.frame_bytes = self.frame_samples * 2 * self.channels

        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_evt = threading.Event()
        self._reader_thread = None
        self._stderr_thread = None
        self._proc = None
        self.queue_overflow_count = 0

    def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        # Read exactly frame_bytes each iteration. On networked SSH streams
        # a single read() may return fewer bytes even when the stream is live.
        # Accumulate until we have a full frame, or exit if the subprocess ends.
        while not self._stop_evt.is_set():
            buf = bytearray()
            bytes_needed = self.frame_bytes
            while bytes_needed > 0 and not self._stop_evt.is_set():
                part = self._proc.stdout.read(bytes_needed)
                if not part:
                    # subprocess closed or EOF
                    buf = None
                    break
                buf.extend(part)
                bytes_needed = self.frame_bytes - len(buf)

            if not buf:
                break

            frame = bytes(buf)
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self.queue_overflow_count += 1
                if self.queue_overflow_count == 1 or self.queue_overflow_count % 25 == 0:
                    print(
                        "[WARN] NAO audio queue overflow "
                        f"count={self.queue_overflow_count} maxsize={self._queue.maxsize}"
                    )
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except Exception:
                    pass

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return

        while not self._stop_evt.is_set():
            line = self._proc.stderr.readline()
            if not line:
                break
            txt = line.decode("utf-8", errors="replace").strip()
            if txt:
                print(f"[NAO-AUDIO] {txt}")

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def read_frame(self, timeout_sec: float = 1.0) -> bytes:
        if self._proc is None:
            raise RuntimeError("NAO audio source is not started")

        while True:
            try:
                return self._queue.get(timeout=timeout_sec)
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"NAO audio stream stopped unexpectedly (exit={self._proc.returncode})"
                    )

    def clear_buffer(self) -> None:
        self._clear_queue()

    def __enter__(self):
        arecord_parts = [
            "arecord",
            "-q",
            "-r",
            str(self.samplerate),
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-t",
            "raw",
        ]
        if self.arecord_device:
            arecord_parts.extend(["-D", self.arecord_device])

        remote_cmd = " ".join(arecord_parts)
        # Allow overriding the ssh target so users can use Host aliases from ~/.ssh/config
        ssh_target = os.environ.get("NAO_SSH_HOST", f"{self.nao_user}@{self.nao_ip}")
        cmd = ["ssh"]

        # Prefer an explicit identity file if provided via env or present at ~/.ssh/nao_ed25519
        key_path = os.environ.get("NAO_SSH_KEY") or os.path.expanduser("~/.ssh/nao_ed25519")
        if key_path and os.path.exists(os.path.expanduser(key_path)):
            cmd.extend(["-i", os.path.expanduser(key_path)])

        cmd.extend([
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "TCPKeepAlive=yes",
            "-o",
            "ConnectTimeout=8",
            ssh_target,
            remote_cmd,
        ])
        print(f"[INFO] NAO audio ssh command: {' '.join(cmd[:-1])} '{ssh_target}'")
        print(f"[INFO] NAO audio remote cmd: {remote_cmd}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

        time.sleep(0.2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                "Failed to start NAO audio source via SSH/arecord "
                f"(exit={self._proc.returncode}). Check SSH access, arecord availability, and device configuration."
            )

        print(
            "[INFO] NAO continuous audio source started "
            f"ip={self.nao_ip} sr={self.samplerate} frame_ms={self.frame_ms}"
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_evt.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._clear_queue()


def run_nao_record_and_transcribe(whisper_model, default_nao_ip: str, default_nao_port: int) -> str:
    """Run remote NAO recording and return transcribed text (or empty string)."""
    nao_user = os.environ.get("NAO_USER", "nao")
    nao_ip_runtime = os.environ.get("NAO_IP", default_nao_ip)
    _nao_port_runtime = os.environ.get("NAO_PORT", str(default_nao_port))
    record_duration = int(os.environ.get("NAO_RECORD_SEC", "5"))
    prep_seconds = int(os.environ.get("NAO_PREP_SEC", "3"))

    print(f"Prepare: you will have {prep_seconds}s to get ready, then NAO will record for {record_duration}s.")
    try:
        for i in range(prep_seconds, 0, -1):
            print(f"Speak in: {i}...", end="\r")
            time.sleep(1)
        print("Go! Recording starting...".ljust(60))
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled prep.")
        return ""

    remote_py = "/home/nao/nao_record.py"
    remote_out = "/home/nao/record.wav"

    ssh_cmd = [
        "ssh",
        f"{nao_user}@{nao_ip_runtime}",
        "/usr/bin/python2 {} --ip {} --duration {} --out {}".format(
            remote_py,
            nao_ip_runtime,
            record_duration,
            remote_out,
        ),
    ]

    print(f"Starting remote recording on NAO {nao_ip_runtime} for {record_duration}s...")
    try:
        subprocess.check_call(ssh_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Remote record failed: {e}")
        print("Tip: ensure /home/nao/nao_record.py exists on the robot and SSH password/key available.")
        return ""

    tmp_local = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_local_path = tmp_local.name
    tmp_local.close()

    scp_cmd = ["scp", f"{nao_user}@{nao_ip_runtime}:{remote_out}", tmp_local_path]
    try:
        subprocess.check_call(scp_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] SCP download failed: {e}")
        try:
            os.remove(tmp_local_path)
        except Exception:
            pass
        return ""

    try:
        user_text = transcribe_file_with_whisper(whisper_model, tmp_local_path)
    except Exception as e:
        print(f"[WARN] STT-from-NAO-file error: {e}")
        user_text = ""

    tests_dir = Path(__file__).resolve().parents[1] / "Tests"
    try:
        if not user_text:
            try:
                tests_dir.mkdir(parents=True, exist_ok=True)
                saved = tests_dir / "nao_last.wav"
                shutil.move(tmp_local_path, str(saved))
                print(f"[DEBUG] Saved NAO recording for inspection: {saved}")
            except Exception:
                try:
                    os.remove(tmp_local_path)
                except Exception:
                    pass
        else:
            try:
                os.remove(tmp_local_path)
            except Exception:
                pass
    except Exception:
        try:
            os.remove(tmp_local_path)
        except Exception:
            pass

    if not user_text:
        print("[WARN] Δεν ανιχνεύτηκε κείμενο στο sample.")
    else:
        print("Καταγραφή (NAO):", user_text)

    return user_text
