import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

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
    import vosk

    VOSK_AVAILABLE = True
except Exception:
    sd = None
    vosk = None
    VOSK_AVAILABLE = False


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


def configure_input_device_from_env() -> None:
    if not VOSK_AVAILABLE or sd is None:
        return

    sounddevice_device = os.environ.get("SOUNDDEVICE_DEVICE")
    if not sounddevice_device:
        return

    try:
        sd.default.device = int(sounddevice_device)
        print(f"[INFO] sounddevice default device set to {sounddevice_device}")
    except Exception as e:
        print(f"[WARN] Invalid SOUNDDEVICE_DEVICE value: {e}")


def transcribe_push_to_talk(model, samplerate: int = 16000, max_seconds: Optional[int] = None) -> str:
    if model is None or not VOSK_AVAILABLE:
        raise RuntimeError("VOSK unavailable")

    try:
        dev_info = sd.query_devices(kind="input")
        device_sr = int(dev_info.get("default_samplerate", samplerate))
    except Exception:
        device_sr = samplerate

    rec = vosk.KaldiRecognizer(model, device_sr)
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
        while not stopper.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                    print(f"[INFO] Push-to-talk: reached max {max_seconds}s, stopping.")
                    stopper.set()
                continue

            rec.AcceptWaveform(data)
            if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                print(f"[INFO] Push-to-talk: reached max {max_seconds}s, stopping.")
                stopper.set()

        while not audio_queue.empty():
            data = audio_queue.get()
            rec.AcceptWaveform(data)

    final = rec.FinalResult()
    try:
        parsed = json.loads(final)
        return parsed.get("text", "").strip()
    except Exception:
        return ""


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


def transcribe_file_with_vosk(model, filepath: str, samplerate: int = 16000) -> str:
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
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
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
