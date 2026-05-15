import subprocess

from nao_common import (
    DEFAULT_TEXT_SYSTEM_PROMPT,
    NAO_IP,
    NAO_PORT,
    OLLAMA_MODEL,
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

# -----------------------------
# Config (all overridable by env)
# -----------------------------
OLLAMA_TEMPERATURE = get_ollama_temperature()
SYSTEM_PROMPT = get_system_prompt(DEFAULT_TEXT_SYSTEM_PROMPT)
OLLAMA_TIMEOUT_SEC = get_ollama_timeout(120)


# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"OLLAMA model={OLLAMA_MODEL}")
    print("Γράψε κάτι (Enter για έξοδο):")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            break

        # 1️⃣ Rules πρώτα
        reply = rule_based_response(user_text)

        # 2️⃣ Αν δεν υπάρχει rule → Ollama
        if reply is None:
            try:
                reply = ollama_generate(
                    user_text,
                    system_prompt=SYSTEM_PROMPT,
                    timeout_sec=OLLAMA_TIMEOUT_SEC,
                    options=build_ollama_options(OLLAMA_TEMPERATURE),
                )
                reply = clean_response(reply)
            except Exception as e:
                print(f"[WARN] Ollama error: {e}")
                continue

        print("NAO:", reply)

        # 3️⃣ Speak
        try:
            nao_say(tts_clip(reply))
        except subprocess.TimeoutExpired:
            print("[WARN] NAO timeout")
        except Exception as e:
            print(f"[WARN] NAO error: {e}")

    print("Τέλος")