import os
import tempfile
import subprocess
import time

from nao_common import (
    DEFAULT_VOICE_SYSTEM_PROMPT,
    NAO_IP,
    NAO_PORT,
    NAO_TTS_TIMEOUT_SEC,
    OLLAMA_MODEL,
    OLLAMA_URL,
    build_ollama_options,
    clean_response,
    get_ollama_temperature,
    get_ollama_timeout,
    get_system_prompt,
    nao_say,
    ollama_generate,
    rule_based_response,
    tts_clip,
)

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import speech_recognition as sr

# -----------------------------
# Config (env overridable)
# -----------------------------
OLLAMA_TEMPERATURE = get_ollama_temperature()
OLLAMA_TIMEOUT_SEC = get_ollama_timeout(180)
SYSTEM_PROMPT = get_system_prompt(DEFAULT_VOICE_SYSTEM_PROMPT)

STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "el-GR")

# recording settings (laptop mic)
REC_SECONDS = float(os.environ.get("REC_SECONDS", "5"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))  # good for STT


def record_wav_to_temp(seconds: float) -> str:
    print(f"Recording for {seconds:.1f}s... speak now")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    audio = np.squeeze(audio)  # shape (N,)
    fd, wav_path = tempfile.mkstemp(prefix="stt_", suffix=".wav")
    os.close(fd)
    wav_write(wav_path, SAMPLE_RATE, audio)
    return wav_path


def stt_google_from_wav(wav_path: str) -> str | None:
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)

    try:
        return r.recognize_google(audio, language=STT_LANGUAGE).strip()
    except sr.UnknownValueError:
        print("[STT] Δεν κατάλαβα. Ξανά.")
        return None
    except sr.RequestError as e:
        print(f"[STT] Network/API error: {e}")
        return None


def main():
    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"Ollama: {OLLAMA_URL} model={OLLAMA_MODEL}")
    print(f"STT: Google (ONLINE) lang={STT_LANGUAGE}")
    print(f"Recording: {REC_SECONDS}s @ {SAMPLE_RATE}Hz")
    print("Say 'αντιο' to quit.")

    try:
        nao_say("Γεια σου! Σε ακούω.")
    except Exception:
        pass

    while True:
        wav_path = None
        try:
            wav_path = record_wav_to_temp(REC_SECONDS)
            text = stt_google_from_wav(wav_path)
            if not text:
                continue

            print("You:", text)

            if text.lower() in ["στοπ", "stop", "τέλος", "exit", "quit"]:
                break

            reply = rule_based_response(text)
            if reply is None:
                reply = ollama_generate(
                    text,
                    system_prompt=SYSTEM_PROMPT,
                    timeout_sec=OLLAMA_TIMEOUT_SEC,
                    options=build_ollama_options(OLLAMA_TEMPERATURE),
                )
                reply = clean_response(reply)

            print("AI:", reply)

            try:
                nao_say(tts_clip(reply))
            except subprocess.TimeoutExpired:
                print(f"[WARN] NAO TTS timeout ({NAO_TTS_TIMEOUT_SEC}s). Continuing...")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] NAO TTS failed (exit={e.returncode}). Continuing...")
            except Exception as e:
                print(f"[WARN] NAO TTS unexpected error: {e}. Continuing...")

            time.sleep(0.2)

        except KeyboardInterrupt:
            break
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    print("Done")


if __name__ == "__main__":
    main()