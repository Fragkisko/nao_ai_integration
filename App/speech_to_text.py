import json
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

try:
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly

    PY_AUDIO_FALLBACK_AVAILABLE = True
except Exception:
    sf = None
    np = None
    resample_poly = None
    PY_AUDIO_FALLBACK_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    import vosk

    VOSK_AVAILABLE = True
except Exception:
    vosk = None
    VOSK_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    WHISPER_AVAILABLE = False


def load_vosk_model(path: str):
    if not VOSK_AVAILABLE:
        return None

    try:
        if not Path(path).exists():
            print(f"[WARN] VOSK model not found at {path}; voice input disabled")
            return None
        print(f"Loading VOSK model from {path} (may take a moment)...")
        return vosk.Model(path)
    except Exception as e:
        print(f"[WARN] VOSK model load error: {e}")
        return None


def load_whisper_model(model_name: str = "small"):
    if not WHISPER_AVAILABLE:
        print("[WARN] faster-whisper is not installed; voice input disabled")
        return None

    try:
        print(f"Loading faster-whisper model '{model_name}' (may take a moment)...")
        # Use FP16 when available; fallback to INT8 on CPU-only systems.
        return WhisperModel(model_name, compute_type="auto")
    except Exception as e:
        print(f"[WARN] faster-whisper model load error: {e}")
        return None


def configure_input_device_from_env() -> None:
    if not SOUNDDEVICE_AVAILABLE or sd is None:
        return

    sounddevice_device = os.environ.get("SOUNDDEVICE_DEVICE")
    if not sounddevice_device:
        return

    try:
        sd.default.device = int(sounddevice_device)
        print(f"[INFO] sounddevice default device set to {sounddevice_device}")
    except Exception as e:
        print(f"[WARN] Invalid SOUNDDEVICE_DEVICE value: {e}")


def resolve_input_device_from_env(*, strict: bool = False) -> Optional[int]:
    """Resolve explicit input device index from env without silent fallback."""
    if not SOUNDDEVICE_AVAILABLE or sd is None:
        if strict:
            raise RuntimeError("sounddevice unavailable")
        return None

    sounddevice_device = os.environ.get("SOUNDDEVICE_DEVICE")
    if sounddevice_device is None or not sounddevice_device.strip():
        return None

    raw = sounddevice_device.strip()
    try:
        index = int(raw)
    except Exception:
        if strict:
            raise RuntimeError(f"Invalid SOUNDDEVICE_DEVICE='{raw}' (must be integer index)")
        print(f"[WARN] Invalid SOUNDDEVICE_DEVICE value: '{raw}'")
        return None

    try:
        dev = sd.query_devices(index, kind="input")
        print(
            "[INFO] Using explicit input device "
            f"index={index} name='{dev.get('name', 'unknown')}' "
            f"native_sr={int(dev.get('default_samplerate', 0) or 0)}"
        )
        return index
    except Exception as e:
        if strict:
            raise RuntimeError(f"Cannot use SOUNDDEVICE_DEVICE={index}: {e}")
        print(f"[WARN] Cannot use SOUNDDEVICE_DEVICE={index}: {e}")
        return None


@dataclass(frozen=True)
class ContinuousVADConfig:
    frame_ms: int = 30
    preroll_ms: int = 300
    min_speech_ms: int = 450
    silence_ms: int = 900
    max_segment_seconds: int = 12
    start_trigger_frames: int = 3
    start_threshold_db: float = 12.0
    stop_threshold_db: float = 6.0
    calibration_seconds: float = 1.0
    target_samplerate: int = 16000


class ContinuousVADCapture:
    """Capture utterances continuously from microphone using an RMS-based VAD."""

    def __init__(
        self,
        config: Optional[ContinuousVADConfig] = None,
        *,
        device_index: Optional[int] = None,
        frame_reader: Optional[Callable[[float], bytes]] = None,
        source_native_samplerate: Optional[int] = None,
        clear_external_buffer: Optional[Callable[[], None]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not PY_AUDIO_FALLBACK_AVAILABLE or np is None or resample_poly is None:
            raise RuntimeError("numpy/scipy are required for continuous VAD mode")
        if frame_reader is None and (not SOUNDDEVICE_AVAILABLE or sd is None):
            raise RuntimeError("sounddevice unavailable")

        self.config = config or ContinuousVADConfig()
        self.logger = logger or print
        self.device_index = device_index
        self.frame_reader = frame_reader
        self.source_native_samplerate = source_native_samplerate
        self.clear_external_buffer = clear_external_buffer

        self.native_samplerate = self.config.target_samplerate
        self.blocksize = max(1, int(self.native_samplerate * self.config.frame_ms / 1000))

        self._audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
        self._capture_enabled = threading.Event()
        self._capture_enabled.set()
        self._stream = None

    def _log(self, event: str, **fields: object) -> None:
        parts = [f"event={event}"]
        for key, val in fields.items():
            parts.append(f"{key}={val}")
        self.logger("[VAD] " + " ".join(parts))

    def _clear_audio_queue(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Exception:
                break

    def set_capture_enabled(self, enabled: bool) -> None:
        if self.frame_reader is not None and self.clear_external_buffer is not None:
            try:
                self.clear_external_buffer()
            except Exception as e:
                self._log("external_buffer_clear_failed", error=str(e))

        if enabled:
            self._capture_enabled.set()
            self._clear_audio_queue()
            self._log("capture_enabled")
            return

        self._capture_enabled.clear()
        self._clear_audio_queue()
        self._log("capture_disabled")

    def _audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            self._log("stream_status", status=str(status))
        if not self._capture_enabled.is_set():
            return

        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            self._log("queue_overflow", maxsize=self._audio_queue.maxsize)

    def __enter__(self):
        if self.frame_reader is not None:
            native_sr = int(self.source_native_samplerate or self.config.target_samplerate)
            if native_sr <= 0:
                raise RuntimeError(f"Invalid source native samplerate: {native_sr}")
            self.native_samplerate = native_sr
            self.blocksize = max(1, int(self.native_samplerate * self.config.frame_ms / 1000))
            self._log(
                "source_opened",
                source="external",
                native_sr=self.native_samplerate,
                blocksize=self.blocksize,
                target_sr=self.config.target_samplerate,
            )
            return self

        if self.device_index is None:
            dev_info = sd.query_devices(kind="input")
            device_for_stream = None
        else:
            dev_info = sd.query_devices(self.device_index, kind="input")
            device_for_stream = self.device_index

        self.native_samplerate = int(dev_info.get("default_samplerate", self.config.target_samplerate))
        if self.native_samplerate <= 0:
            raise RuntimeError(f"Invalid native samplerate from input device: {self.native_samplerate}")

        self.blocksize = max(1, int(self.native_samplerate * self.config.frame_ms / 1000))
        self._stream = sd.RawInputStream(
            samplerate=self.native_samplerate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
            device=device_for_stream,
        )

        try:
            self._stream.start()
        except Exception as e:
            msg = (
                "Failed to open input stream "
                f"device={self.device_index if self.device_index is not None else 'default'} "
                f"native_sr={self.native_samplerate} blocksize={self.blocksize}: {e}"
            )
            self._log("stream_open_failed", error=msg)
            raise RuntimeError(msg)

        self._log(
            "stream_opened",
            device=self.device_index if self.device_index is not None else "default",
            native_sr=self.native_samplerate,
            blocksize=self.blocksize,
            target_sr=self.config.target_samplerate,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._log("stream_close_failed", error=str(e))
            finally:
                self._stream = None
                self._log("stream_closed")

    def _read_next_frame(self, timeout_sec: float) -> bytes:
        if self.frame_reader is not None:
            return self.frame_reader(timeout_sec)
        return self._audio_queue.get(timeout=timeout_sec)

    @staticmethod
    def _rms_dbfs_from_pcm16(frame_bytes: bytes) -> float:
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        if arr.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(arr * arr))) / 32768.0
        rms = max(rms, 1e-9)
        return 20.0 * math.log10(rms)

    def _resample_pcm16(self, pcm_bytes: bytes, src_sr: int) -> bytes:
        if src_sr == self.config.target_samplerate:
            return pcm_bytes

        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return pcm_bytes

        resampled = resample_poly(arr, self.config.target_samplerate, src_sr)
        clipped = np.clip(resampled, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    def next_segment(self) -> Dict[str, object]:
        """Block until an utterance segment is detected and return audio + metrics."""
        frame_ms = self.config.frame_ms
        max_preroll_frames = max(1, int(self.config.preroll_ms / frame_ms))
        start_trigger_frames = max(1, self.config.start_trigger_frames)
        min_speech_frames = max(1, int(self.config.min_speech_ms / frame_ms))
        silence_stop_frames = max(1, int(self.config.silence_ms / frame_ms))
        max_total_frames = max(1, int(self.config.max_segment_seconds * 1000 / frame_ms))
        calibration_frames = max(1, int(self.config.calibration_seconds * 1000 / frame_ms))

        preroll = deque(maxlen=max_preroll_frames)
        segment_frames: List[bytes] = []

        noise_floor_db = -55.0
        calibration_buf: List[float] = []
        speech_trigger_count = 0
        speech_frame_count = 0
        silence_frame_count = 0
        total_frame_count = 0
        in_speech_total_frames = 0
        in_speech = False
        segment_start_ts = 0.0
        end_reason = "unknown"

        self._log("listening", noise_floor_db=round(noise_floor_db, 2))

        while True:
            try:
                frame = self._read_next_frame(timeout_sec=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                self._log("source_read_failed", error=str(e), in_speech=in_speech)
                raise RuntimeError(f"audio frame read failed: {e}")
            dbfs = self._rms_dbfs_from_pcm16(frame)
            total_frame_count += 1

            if len(calibration_buf) < calibration_frames:
                calibration_buf.append(dbfs)
                if len(calibration_buf) == calibration_frames:
                    calibration_sorted = sorted(calibration_buf)
                    idx = max(0, int(0.8 * len(calibration_sorted)) - 1)
                    noise_floor_db = calibration_sorted[idx]
                    self._log("noise_calibrated", noise_floor_db=round(noise_floor_db, 2))
                    # Drop calibration preroll so the first segment starts with fresh context.
                    preroll.clear()

            calibrated = len(calibration_buf) >= calibration_frames
            if not calibrated:
                preroll.append(frame)
                continue

            start_threshold = noise_floor_db + float(self.config.start_threshold_db)
            stop_threshold = noise_floor_db + float(self.config.stop_threshold_db)

            if not in_speech:
                preroll.append(frame)
                if dbfs >= start_threshold:
                    speech_trigger_count += 1
                else:
                    speech_trigger_count = 0
                    if dbfs < noise_floor_db + 3.0:
                        noise_floor_db = 0.98 * noise_floor_db + 0.02 * dbfs

                if speech_trigger_count >= start_trigger_frames:
                    in_speech = True
                    segment_start_ts = time.time()
                    segment_frames.extend(list(preroll))
                    speech_frame_count = len(preroll)
                    in_speech_total_frames = len(preroll)
                    silence_frame_count = 0
                    self._log(
                        "speech_started",
                        threshold_db=round(start_threshold, 2),
                        frame_count=speech_trigger_count,
                    )
                continue

            segment_frames.append(frame)
            in_speech_total_frames += 1
            if dbfs >= stop_threshold:
                speech_frame_count += 1
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            if silence_frame_count >= silence_stop_frames:
                end_reason = "silence"
                break

            if in_speech_total_frames >= max_total_frames:
                end_reason = "max_duration"
                break

        elapsed = max(0.0, time.time() - segment_start_ts)
        speech_ms = int(speech_frame_count * frame_ms)
        raw_pcm = b"".join(segment_frames)
        pcm = self._resample_pcm16(raw_pcm, self.native_samplerate)

        metrics: Dict[str, object] = {
            "duration_ms": int(elapsed * 1000),
            "speech_ms": speech_ms,
            "segment_ms": int(len(segment_frames) * frame_ms),
            "frames": len(segment_frames),
            "native_sr": self.native_samplerate,
            "target_sr": self.config.target_samplerate,
            "noise_floor_db": round(noise_floor_db, 2),
            "end_reason": end_reason,
        }

        if speech_frame_count < min_speech_frames:
            self._log("segment_discarded", reason="too_short", speech_ms=speech_ms, end_reason=end_reason)
            return {
                "discarded": True,
                "audio_pcm16": b"",
                "samplerate": self.config.target_samplerate,
                "metrics": metrics,
            }

        self._log(
            "segment_ready",
            speech_ms=speech_ms,
            segment_ms=metrics["segment_ms"],
            end_reason=end_reason,
        )
        return {
            "discarded": False,
            "audio_pcm16": pcm,
            "samplerate": self.config.target_samplerate,
            "metrics": metrics,
        }


def transcribe_pcm16_with_whisper(
    model,
    audio_pcm16: bytes,
    samplerate: int = 16000,
    language: Optional[str] = None,
) -> str:
    if model is None or not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper STT unavailable")
    if not audio_pcm16:
        return ""

    tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmpf.name
    tmpf.close()
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_pcm16)
        return transcribe_file_with_whisper(model, tmp_path, samplerate=samplerate, language=language)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _normalize_vocab_entries(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in items:
        token = (raw or "").strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_vocab_file_entries(vocab_file: str) -> List[str]:
    path = Path(vocab_file)
    if not path.exists():
        print(f"[WARN] STT vocab file not found: {vocab_file}")
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Cannot read STT vocab file {vocab_file}: {e}")
        return []

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict) and isinstance(data.get("phrases"), list):
                return [str(x) for x in data["phrases"]]
            print("[WARN] JSON vocab file must be an array or an object with a 'phrases' array")
            return []
        except Exception as e:
            print(f"[WARN] Invalid JSON vocab file {vocab_file}: {e}")
            return []

    entries: List[str] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        entries.extend(parts)
    return entries


def _resolve_vocab_files_from_env() -> List[Path]:
    base_dir = Path(__file__).resolve().parent
    vocab_dir_raw = os.environ.get("STT_VOCAB_DIR", "").strip()
    profile_raw = os.environ.get("STT_VOCAB_PROFILE", "all").strip().lower()

    profile_map = {
        "all": ["common.txt", "professors.txt", "courses.txt"],
        "professor": ["common.txt", "professors.txt"],
        "professors": ["common.txt", "professors.txt"],
        "course": ["common.txt", "courses.txt"],
        "courses": ["common.txt", "courses.txt"],
        "common": ["common.txt"],
    }

    selected_profile = profile_raw if profile_raw in profile_map else "all"
    if profile_raw and profile_raw not in profile_map:
        print(f"[WARN] Unknown STT_VOCAB_PROFILE='{profile_raw}', using 'all'")

    resolved_files: List[Path] = []
    if vocab_dir_raw:
        vocab_dir = Path(vocab_dir_raw)
        if not vocab_dir.is_absolute():
            vocab_dir = (base_dir / vocab_dir).resolve()
        for name in profile_map[selected_profile]:
            resolved_files.append(vocab_dir / name)

    legacy_vocab_file = os.environ.get("STT_VOCAB_FILE", "").strip()
    if legacy_vocab_file:
        resolved_files.append(Path(legacy_vocab_file))

    return resolved_files


def load_custom_vocabulary_from_env() -> List[str]:
    """Load optional STT vocabulary from env vars for Vosk grammar injection.

    Supported env vars:
    - STT_VOCAB: comma-separated words/phrases.
    - STT_VOCAB_DIR: directory containing common.txt/professors.txt/courses.txt.
    - STT_VOCAB_PROFILE: all|professor|course|common.
    - STT_VOCAB_FILE: .txt or .json file with extra vocabulary.
    """
    inline_raw = os.environ.get("STT_VOCAB", "")

    inline_entries = [p.strip() for p in inline_raw.split(",")] if inline_raw else []
    file_entries: List[str] = []
    for vocab_path in _resolve_vocab_files_from_env():
        file_entries.extend(_load_vocab_file_entries(str(vocab_path)))

    return _normalize_vocab_entries([*inline_entries, *file_entries])


def _build_vosk_recognizer(model, samplerate: int, custom_vocabulary: Optional[List[str]] = None):
    vocab = _normalize_vocab_entries(custom_vocabulary or load_custom_vocabulary_from_env())
    if not vocab:
        return vosk.KaldiRecognizer(model, samplerate)

    # Keep [unk] so recognizer can still emit unseen words.
    grammar_words = _normalize_vocab_entries([*vocab, "[unk]"])
    try:
        grammar_json = json.dumps(grammar_words, ensure_ascii=False)
        print(f"[INFO] STT vocabulary injection active ({len(vocab)} entries)")
        return vosk.KaldiRecognizer(model, samplerate, grammar_json)
    except Exception as e:
        print(f"[WARN] Failed to apply STT vocabulary injection: {e}")
        return vosk.KaldiRecognizer(model, samplerate)


def transcribe_push_to_talk(
    model,
    samplerate: int = 16000,
    max_seconds: Optional[int] = None,
    custom_vocabulary: Optional[List[str]] = None,
) -> str:
    del custom_vocabulary  # Not used by faster-whisper.
    if model is None or not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper STT unavailable")
    if not SOUNDDEVICE_AVAILABLE or sd is None:
        raise RuntimeError("sounddevice unavailable")

    try:
        dev_info = sd.query_devices(kind="input")
        device_sr = int(dev_info.get("default_samplerate", samplerate))
    except Exception:
        device_sr = samplerate

    audio_queue: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        audio_queue.put(bytes(indata))

    print("Push-to-talk: Πάτα Enter για να ξεκινήσει η εγγραφή, ξαναπάτα Enter για να σταματήσει.")
    if max_seconds is not None:
        print(f"Σημείωση: Η εγγραφή θα σταματήσει αυτόματα μετά από {max_seconds} δευτερόλεπτα.")
    input("Πάτα Enter για εκκίνηση εγγραφής...")

    with sd.RawInputStream(
        samplerate=device_sr,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Recording... Πάτα Enter για στάση.")
        stopper = threading.Event()

        def stopper_thread():
            input()
            stopper.set()

        thread = threading.Thread(target=stopper_thread, daemon=True)
        thread.start()

        start_ts = time.time()
        captured_frames: List[bytes] = []
        while not stopper.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                    print(f"[INFO] Push-to-talk: reached max {max_seconds}s, stopping.")
                    stopper.set()
                continue

            captured_frames.append(data)
            if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                print(f"[INFO] Push-to-talk: reached max {max_seconds}s, stopping.")
                stopper.set()

        while not audio_queue.empty():
            captured_frames.append(audio_queue.get())

    if not captured_frames:
        return ""

    tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmpf.name
    tmpf.close()
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(device_sr)
            wf.writeframes(b"".join(captured_frames))

        return transcribe_file_with_whisper(model, tmp_path, samplerate=samplerate)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def transcribe_file_with_whisper(
    model,
    filepath: str,
    samplerate: int = 16000,
    language: Optional[str] = None,
) -> str:
    if model is None or not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper STT unavailable")

    chosen_language = (language or os.environ.get("STT_LANGUAGE", "el")).strip() or None

    try:
        ok_path = _ensure_wav_from_any(filepath, target_sr=samplerate)
    except Exception as e:
        try:
            subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok_path = _ffmpeg_resample_if_needed(filepath, target_sr=samplerate)
        except Exception:
            print(f"[WARN] cannot transcode file: {e}")
            return ""

    cleanup_path = ok_path if ok_path != filepath else None

    try:
        segments, _ = model.transcribe(
            ok_path,
            language=chosen_language,
            vad_filter=True,
        )
        return " ".join((seg.text or "").strip() for seg in segments if (seg.text or "").strip()).strip()
    except Exception as e:
        print(f"[WARN] Whisper transcription failed: {e}")
        return ""
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except Exception:
                pass


def _ffmpeg_resample_if_needed(src_path: str, target_sr: int = 16000) -> str:
    """Ensure a WAV file at target sample rate/mono and return converted path."""
    with wave.open(src_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()

    if sr == target_sr and ch == 1:
        return src_path

    tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmpf.name
    tmpf.close()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        tmp_path,
    ]
    try:
        subprocess.check_call(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_path
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg resample failed: {e}")


def _ensure_wav_from_any(src_path: str, target_sr: int = 16000) -> str:
    """Read common formats and convert to mono 16k WAV if needed."""
    try:
        with wave.open(src_path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            if sr == target_sr and ch == 1:
                return src_path
    except Exception:
        pass

    if not PY_AUDIO_FALLBACK_AVAILABLE:
        raise RuntimeError(
            "File is not a plain WAV and Python fallback libs are not installed. "
            "Install ffmpeg or run: pip install soundfile numpy scipy"
        )

    try:
        data, sr = sf.read(src_path, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"soundfile cannot read input file: {e}")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        try:
            data = resample_poly(data, target_sr, sr)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(f"resampling failed: {e}")

    tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmpf.name
    tmpf.close()
    try:
        max_abs = np.max(np.abs(data)) if data.size > 0 else 1.0
        if max_abs > 0:
            normalized = data / max_abs
        else:
            normalized = data
        int16_data = (normalized * 32767).astype("int16")
        sf.write(tmp_path, int16_data, sr, subtype="PCM_16")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"cannot write tmp wav: {e}")

    return tmp_path


def transcribe_file_with_vosk(
    model,
    filepath: str,
    samplerate: int = 16000,
    custom_vocabulary: Optional[List[str]] = None,
) -> str:
    if model is None or not VOSK_AVAILABLE:
        raise RuntimeError("VOSK unavailable")

    try:
        ok_path = _ensure_wav_from_any(filepath, target_sr=samplerate)
    except Exception as e:
        try:
            subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok_path = _ffmpeg_resample_if_needed(filepath, target_sr=samplerate)
        except Exception:
            print(f"[WARN] cannot transcode file: {e}")
            return ""

    cleanup_path = ok_path if ok_path != filepath else None

    try:
        with wave.open(ok_path, "rb") as wf:
            rec = _build_vosk_recognizer(model, wf.getframerate(), custom_vocabulary=custom_vocabulary)
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        final = rec.FinalResult()
        try:
            parsed = json.loads(final)
            return parsed.get("text", "").strip()
        except Exception:
            return ""
    except Exception as e:
        print(f"[WARN] cannot open wav file: {e}")
        return ""
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except Exception:
                pass
