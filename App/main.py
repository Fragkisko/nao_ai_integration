import subprocess
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
    VOSK_AVAILABLE,
    configure_input_device_from_env,
    load_vosk_model,
    transcribe_push_to_talk,
)


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

    vosk_available = VOSK_AVAILABLE
    vosk_model = None
    if vosk_available:
        vosk_model = load_vosk_model(settings.vosk_model_path)
        if vosk_model is None:
            vosk_available = False

    if vosk_available:
        configure_input_device_from_env()

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
            if not vosk_available or vosk_model is None:
                print("[WARN] Φωνητική είσοδος μη διαθέσιμη — επίλεξε πληκτρολόγιο.")
                continue
            try:
                user_text = transcribe_push_to_talk(vosk_model, max_seconds=settings.push_max_seconds)
            except Exception as e:
                print(f"[WARN] STT error: {e}")
                continue
            if not user_text:
                print("[WARN] Δεν ανιχνεύτηκε κείμενο στην εγγραφή.")
                continue
            print("Καταγραφή:", user_text)
        elif mode == "n":
            user_text = run_nao_record_and_transcribe(vosk_model, NAO_IP, NAO_PORT)
            if not user_text:
                continue
        else:
            print("Άγνωστη επιλογή — πάτα k, v, n ή q")
            continue

        rag_result = answer_query(rag_engine, user_text)

        if settings.rag_debug_chunks and rag_module is not None:
            for i, item in enumerate(rag_result.get("chunks", []), start=1):
                print(rag_module.format_chunk(item, i))

        reply = (rag_result.get("answer") or "").strip()
        if not reply:
            if rag_result.get("error"):
                print(f"[WARN] {rag_result['error']}")
            reply = settings.rag_fallback_reply

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