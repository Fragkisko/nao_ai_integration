import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterable, List, Optional

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
    if model is None or not VOSK_AVAILABLE:
        raise RuntimeError("VOSK unavailable")

    try:
        dev_info = sd.query_devices(kind="input")
        device_sr = int(dev_info.get("default_samplerate", samplerate))
    except Exception:
        device_sr = samplerate

    rec = _build_vosk_recognizer(model, device_sr, custom_vocabulary=custom_vocabulary)
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
