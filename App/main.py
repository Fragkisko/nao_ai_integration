import importlib.util
import os
import argparse
from pathlib import Path
import subprocess
import queue
import threading
import json
import time
import shutil
import tempfile
import wave
import sys
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

# Optional offline STT (VOSK)
try:
    import sounddevice as sd
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    sd = None
    vosk = None
    VOSK_AVAILABLE = False

# Load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[1] / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

# Parse CLI args early so we can override env vars before importing nao_common
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nao-ip", help="Override NAO IP address")
parser.add_argument("--nao-port", type=int, help="Override NAO port")
args, _ = parser.parse_known_args()
if args.nao_ip:
    os.environ["NAO_IP"] = args.nao_ip
if args.nao_port:
    os.environ["NAO_PORT"] = str(args.nao_port)

from nao_common import (
    NAO_IP,
    NAO_PORT,
    nao_say,
    tts_clip,
)

# -----------------------------
# Config (all overridable by env)
# -----------------------------
RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
RAG_DEBUG_CHUNKS = os.environ.get("RAG_DEBUG_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
RAG_FALLBACK_REPLY = "Δεν βρήκα αξιόπιστη απάντηση τώρα"
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "vosk-model-gr"))


def _load_rag_cli_module():
    rag_cli_path = Path(__file__).resolve().parents[1] / "Rag-Chabot" / "Generator" / "rag_cli.py"
    spec = importlib.util.spec_from_file_location("rag_cli_runtime", str(rag_cli_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rag_cli module from {rag_cli_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_vosk_model(path: str):
    if not VOSK_AVAILABLE:
        return None
    try:
        if not Path(path).exists():
            print(f"[WARN] VOSK model not found at {path}; voice input disabled")
            return None
        print(f"Loading VOSK model from {path} (may take a moment)...")
        model = vosk.Model(path)
        return model
    except Exception as e:
        print(f"[WARN] VOSK model load error: {e}")
        return None


def transcribe_push_to_talk(model, samplerate=16000):
    if model is None or not VOSK_AVAILABLE:
        raise RuntimeError("VOSK unavailable")

    # Try to choose a samplerate supported by the default input device
    try:
        dev_info = sd.query_devices(kind='input')
        device_sr = int(dev_info.get('default_samplerate', samplerate))
    except Exception:
        device_sr = samplerate

    rec = vosk.KaldiRecognizer(model, device_sr)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # minor audio device warnings
            pass
        q.put(bytes(indata))

    print("Push-to-talk: Πάτα Enter για να ξεκινήσει η εγγραφή, ξαναπάτα Enter για να σταματήσει.")
    input("Πάτα Enter για εκκίνηση εγγραφής...")

    with sd.RawInputStream(samplerate=device_sr, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("Recording... Πάτα Enter για στάση.")
        stopper = threading.Event()

        def stopper_thread():
            input()
            stopper.set()

        t = threading.Thread(target=stopper_thread, daemon=True)
        t.start()

        while not stopper.is_set():
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue
            rec.AcceptWaveform(data)

        # drain remaining audio
        while not q.empty():
            data = q.get()
            rec.AcceptWaveform(data)

    final = rec.FinalResult()
    try:
        parsed = json.loads(final)
        return parsed.get('text', '').strip()
    except Exception:
        return ''


def _ffmpeg_resample_if_needed(src_path: str, target_sr: int = 16000) -> str:
    """Ensure a WAV file at target sample rate/mono. Returns path to ok file.
    Uses ffmpeg if conversion is necessary. If ffmpeg not found, raises.
    """
    try:
        with wave.open(src_path, 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
    except Exception:
        raise

    if sr == target_sr and ch == 1:
        return src_path

    tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmpf.name
    tmpf.close()

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', src_path,
        '-ar', str(target_sr), '-ac', '1', tmp_path
    ]
    try:
        subprocess.check_call(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_path
    except Exception as e:
        # cleanup and re-raise so caller can fallback or report
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg resample failed: {e}")


def _ensure_wav_from_any(src_path: str, target_sr: int = 16000) -> str:
    """Read many audio formats using Python libs and produce mono 16k WAV.
    Returns path to WAV file ready for VOSK. Raises RuntimeError with guidance if not possible.
    """
    # If it's already a WAV file, try fast path
    try:
        with wave.open(src_path, 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            if sr == target_sr and ch == 1:
                return src_path
    except Exception:
        # not a simple WAV or unreadable by wave; fall through to soundfile
        pass

    if not PY_AUDIO_FALLBACK_AVAILABLE:
        raise RuntimeError(
            "File is not a plain WAV and Python fallback libs are not installed. "
            "Install ffmpeg or run: pip install soundfile numpy scipy"
        )

    try:
        data, sr = sf.read(src_path, dtype='float32')
    except Exception as e:
        raise RuntimeError(f"soundfile cannot read input file: {e}")

    # Convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            # resample_poly is efficient and avoids artifacts
            data = resample_poly(data, target_sr, sr)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(f"resampling failed: {e}")

    # Write to temporary WAV 16-bit PCM
    tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmpf.name
    tmpf.close()
    try:
        # scale float32 -> int16
        max_abs = np.max(np.abs(data)) if data.size > 0 else 1.0
        if max_abs > 0:
            norm = data / max_abs
        else:
            norm = data
        int16 = (norm * 32767).astype('int16')
        sf.write(tmp_path, int16, sr, subtype='PCM_16')
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

    # Ensure WAV is mono 16k using Python fallback first
    try:
        ok_path = _ensure_wav_from_any(filepath, target_sr=samplerate)
    except Exception as e:
        # try ffmpeg as a last resort if available
        try:
            subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok_path = _ffmpeg_resample_if_needed(filepath, target_sr=samplerate)
        except Exception:
            print(f"[WARN] cannot transcode file: {e}")
            return ''

    try:
        wf = wave.open(ok_path, 'rb')
    except Exception as e:
        print(f"[WARN] cannot open wav file: {e}")
        return ''

    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    final = rec.FinalResult()
    try:
        parsed = json.loads(final)
        return parsed.get('text', '').strip()
    except Exception:
        return ''


# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    rag_module = None
    rag_engine = None
    try:
        rag_module = _load_rag_cli_module()
        rag_engine = rag_module.RAGEngine(
            out="../Embedding/out",
            model=RAG_MODEL,
            quiet=True,
            k=RAG_TOP_K,
            structured_mode="llm",
        )
    except Exception as e:
        print(f"[WARN] RAG init error: {e}")

    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"RAG model={RAG_MODEL}")
    print(f"RAG debug chunks={RAG_DEBUG_CHUNKS}")
    


    # Load VOSK model (if available)
    vosk_model = None
    if VOSK_AVAILABLE:
        vosk_model = _load_vosk_model(VOSK_MODEL_PATH)
        if vosk_model is None:
            VOSK_AVAILABLE = False

    # Optionally set sounddevice default device from env
    SOUNDDEVICE_DEVICE = os.environ.get("SOUNDDEVICE_DEVICE")
    if SOUNDDEVICE_DEVICE and VOSK_AVAILABLE and sd is not None:
        try:
            sd.default.device = int(SOUNDDEVICE_DEVICE)
            print(f"[INFO] sounddevice default device set to {SOUNDDEVICE_DEVICE}")
        except Exception as e:
            print(f"[WARN] Invalid SOUNDDEVICE_DEVICE value: {e}")

    print("Επιλογές εισόδου: (k) πληκτρολόγιο / (v) φωνή / (n) NAO (test) / (q) έξοδος")

    while True:
        try:
            mode = input("Επιλογή > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not mode or mode == "q":
            break

        if mode == "k":
            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
        elif mode == "v":
            if not VOSK_AVAILABLE or vosk_model is None:
                print("[WARN] Φωνητική είσοδος μη διαθέσιμη — επίλεξε πληκτρολόγιο.")
                continue
            try:
                user_text = transcribe_push_to_talk(vosk_model)
            except Exception as e:
                print(f"[WARN] STT error: {e}")
                continue
            if not user_text:
                print("[WARN] Δεν ανιχνεύτηκε κείμενο στην εγγραφή.")
                continue
            print("Καταγραφή:", user_text)
        elif mode == "n":
            # Remote NAO recording flow: local prep countdown -> SSH -> run helper -> SCP -> transcribe
            NAO_USER = os.environ.get("NAO_USER", "nao")
            NAO_IP_RUNTIME = os.environ.get("NAO_IP", NAO_IP)
            NAO_PORT_RUNTIME = os.environ.get("NAO_PORT", str(NAO_PORT))
            RECORD_DURATION = int(os.environ.get("NAO_RECORD_SEC", "5"))
            PREP_SECONDS = int(os.environ.get("NAO_PREP_SEC", "3"))

            print(f"Prepare: you will have {PREP_SECONDS}s to get ready, then NAO will record for {RECORD_DURATION}s.")
            try:
                for i in range(PREP_SECONDS, 0, -1):
                    print(f"Speak in: {i}...", end='\r')
                    time.sleep(1)
                print('Go! Recording starting...'.ljust(60))
            except (KeyboardInterrupt, EOFError):
                print('\nCancelled prep.')
                continue

            remote_py = "/home/nao/nao_record.py"
            remote_out = "/home/nao/record.wav"

            ssh_cmd = [
                "ssh",
                f"{NAO_USER}@{NAO_IP_RUNTIME}",
                "/usr/bin/python2 {} --ip {} --duration {} --out {}".format(remote_py, NAO_IP_RUNTIME, RECORD_DURATION, remote_out),
            ]
            print(f"Starting remote recording on NAO {NAO_IP_RUNTIME} for {RECORD_DURATION}s...")
            try:
                subprocess.check_call(ssh_cmd)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Remote record failed: {e}")
                print("Tip: ensure /home/nao/nao_record.py exists on the robot and SSH password/key available.")
                continue

            # Download recorded file to a temp location
            tmp_local = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_local_path = tmp_local.name
            tmp_local.close()
            scp_cmd = ["scp", f"{NAO_USER}@{NAO_IP_RUNTIME}:{remote_out}", tmp_local_path]
            try:
                subprocess.check_call(scp_cmd)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] SCP download failed: {e}")
                try:
                    os.remove(tmp_local_path)
                except Exception:
                    pass
                continue

            # Ensure format and transcribe
            try:
                user_text = transcribe_file_with_vosk(vosk_model, tmp_local_path)
            except Exception as e:
                print(f"[WARN] STT-from-NAO-file error: {e}")
                user_text = ''

            # If transcription is empty, keep the WAV for debugging under ./Tests
            tests_dir = Path(__file__).resolve().parents[1] / 'Tests'
            try:
                if not user_text:
                    try:
                        tests_dir.mkdir(parents=True, exist_ok=True)
                        saved = tests_dir / 'nao_last.wav'
                        shutil.move(tmp_local_path, str(saved))
                        print(f"[DEBUG] Saved NAO recording for inspection: {saved}")
                    except Exception:
                        # fallback: try to remove if move failed
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
                # ensure we don't crash the main loop
                try:
                    os.remove(tmp_local_path)
                except Exception:
                    pass

            if not user_text:
                print("[WARN] Δεν ανιχνεύτηκε κείμενο στο sample.")
                continue

            print("Καταγραφή (NAO):", user_text)

        else:
            print("Άγνωστη επιλογή — πάτα k, v, n ή q")
            continue

        rag_result = {"answer": None, "error": "RAG engine unavailable", "chunks": []}
        if rag_engine is not None:
            try:
                rag_result = rag_engine.answer_query(user_text)
            except Exception as e:
                rag_result = {"answer": None, "error": f"RAG runtime error: {e}", "chunks": []}

        if RAG_DEBUG_CHUNKS and rag_module is not None:
            for i, item in enumerate(rag_result.get("chunks", []), start=1):
                print(rag_module.format_chunk(item, i))

        reply = (rag_result.get("answer") or "").strip()
        if not reply:
            if rag_result.get("error"):
                print(f"[WARN] {rag_result['error']}")
            reply = RAG_FALLBACK_REPLY

        print("NAO:", reply)

        try:
            nao_say(tts_clip(reply))
        except subprocess.TimeoutExpired:
            print("[WARN] NAO timeout")
        except Exception as e:
            print(f"[WARN] NAO error: {e}")

    print("Τέλος")