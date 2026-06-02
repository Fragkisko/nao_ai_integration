import subprocess
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from settings import apply_cli_overrides, build_settings, load_dotenv_from_project_root

load_dotenv_from_project_root()
apply_cli_overrides()

from nao_common import (
    NAO_IP,
    NAO_PORT,
    nao_say,
    tts_clip,
)
from nao_service import run_nao_record_and_transcribe
from rag_service import answer_query, init_rag_engine
from speech_to_text import (
    WHISPER_AVAILABLE,
    VOSK_AVAILABLE,
    configure_input_device_from_env,
    load_whisper_model,
    load_vosk_model,
    transcribe_push_to_talk,
)


def _append_stt_debug_row(settings, *, mode: str, stt_text: str, rag_answer: str = "", rag_error: str = "") -> None:
    if not settings.stt_debug_log_enabled:
        return

    log_path = Path(settings.stt_debug_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8", newline="") as fp:
            writer = DictWriter(
                fp,
                fieldnames=[
                    "timestamp",
                    "mode",
                    "stt_text",
                    "rag_answer",
                    "rag_error",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": mode,
                    "stt_text": stt_text,
                    "rag_answer": rag_answer,
                    "rag_error": rag_error,
                }
            )
    except Exception as e:
        print(f"[WARN] STT debug log write failed: {e}")


def main():
    settings = build_settings()
    rag_module, rag_engine = init_rag_engine(
        settings.rag_model,
        settings.rag_top_k,
        preload_retrieval_on_boot=settings.rag_preload_retrieval_on_boot,
    )

    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"RAG model={settings.rag_model}")
    print(f"RAG preload retrieval on boot={settings.rag_preload_retrieval_on_boot}")
    print(f"RAG debug chunks={settings.rag_debug_chunks}")
    print(f"STT whisper model={settings.stt_whisper_model}")
    print(f"STT debug log enabled={settings.stt_debug_log_enabled}")
    if settings.stt_debug_log_enabled:
        print(f"STT debug log path={settings.stt_debug_log_path}")

    whisper_available = WHISPER_AVAILABLE
    whisper_model = None
    if whisper_available:
        whisper_model = load_whisper_model(settings.stt_whisper_model)
        if whisper_model is None:
            whisper_available = False

    vosk_available = VOSK_AVAILABLE
    vosk_model = None
    if vosk_available:
        vosk_model = load_vosk_model(settings.vosk_model_path)
        if vosk_model is None:
            vosk_available = False

    if whisper_available:
        configure_input_device_from_env()

    print("Επιλογές εισόδου: (k) πληκτρολόγιο / (v) φωνή / (vs) φωνή μόνο STT (debug) / (n) NAO (test) / (q) έξοδος")

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
        elif mode in {"v", "vs"}:
            if not whisper_available or whisper_model is None:
                print("[WARN] Φωνητική είσοδος μη διαθέσιμη — επίλεξε πληκτρολόγιο.")
                continue
            try:
                user_text = transcribe_push_to_talk(whisper_model, max_seconds=settings.push_max_seconds)
            except Exception as e:
                print(f"[WARN] STT error: {e}")
                continue
            if not user_text:
                print("[WARN] Δεν ανιχνεύτηκε κείμενο στην εγγραφή.")
                continue
            print("Καταγραφή:", user_text)
            if mode == "vs":
                _append_stt_debug_row(settings, mode=mode, stt_text=user_text)
                print("[DEBUG] STT-only mode: παράλειψη RAG/AI απάντησης.")
                continue
        elif mode == "n":
            if not whisper_available or whisper_model is None:
                print("[WARN] NAO mode χρειάζεται Whisper STT αλλά δεν είναι διαθέσιμο.")
                continue
            user_text = run_nao_record_and_transcribe(whisper_model, NAO_IP, NAO_PORT)
            if not user_text:
                continue
        else:
            print("Άγνωστη επιλογή — πάτα k, v, vs, n ή q")
            continue

        rag_result = answer_query(rag_engine, user_text)

        if settings.rag_debug_chunks and rag_module is not None:
            for i, item in enumerate(rag_result.get("chunks", []), start=1):
                print(rag_module.format_chunk(item, i))

        reply = (rag_result.get("answer") or "").strip()
        rag_error = rag_result.get("error") or ""
        if not reply:
            if rag_error:
                print(f"[WARN] {rag_error}")
            reply = settings.rag_fallback_reply

        if mode in {"v", "vs"}:
            _append_stt_debug_row(
                settings,
                mode=mode,
                stt_text=user_text,
                rag_answer=reply,
                rag_error=rag_error,
            )

        print("NAO:", reply)

        try:
            nao_say(tts_clip(reply))
        except subprocess.TimeoutExpired:
            print("[WARN] NAO timeout")
        except Exception as e:
            print(f"[WARN] NAO error: {e}")

    print("Τέλος")


if __name__ == "__main__":
    main()