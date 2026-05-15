# -*- coding: utf-8 -*-
import argparse
import time
import json
import urllib2

from naoqi import ALProxy

def ollama_generate(url, model, system, prompt, timeout=60):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9},
    }
    req = urllib2.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib2.urlopen(req, timeout=timeout)
    out = json.loads(resp.read().decode("utf-8"))
    return (out.get("response") or "").strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True)
    p.add_argument("--port", type=int, default=9559)
    p.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    p.add_argument("--ollama_model", default="llama3:latest")
    args = p.parse_args()

    # NAO proxies
    tts = ALProxy("ALTextToSpeech", args.ip, args.port)
    asr = ALProxy("ALSpeechRecognition", args.ip, args.port)
    mem = ALProxy("ALMemory", args.ip, args.port)

    # Try set Greek TTS language if available
    try:
        tts.setLanguage("Greek")
    except Exception:
        pass

    system = (
        u"Μίλα ΜΟΝΟ με σωστά Νέα Ελληνικά. "
        u"Απάντα πολύ σύντομα (1 πρόταση), χωρίς markdown."
    )

    # IMPORTANT: ALSpeechRecognition is often vocabulary-based, not free dictation.
    # Put a starter vocabulary; you can expand.
    vocabulary = [
        u"γεια", u"τι κανεις", u"πως εισαι", u"ονομα", u"ποιος εισαι",
        u"ευχαριστω", u"αντιο", u"στοπ"
    ]

    # Configure ASR
    asr.pause(True)
    asr.setLanguage("Greek")  # if supported
    asr.setVocabulary(vocabulary, False)  # False = no word spotting (stricter)
    asr.pause(False)

    # Subscribe
    subscriber = "nao_asr_dialog"
    try:
        asr.unsubscribe(subscriber)
    except Exception:
        pass
    asr.subscribe(subscriber)

    tts.say("Γεια σου! Σε ακούω. Πες στοπ για να σταματήσω.")

    last_text = None
    last_time = 0

    try:
        while True:
            data = mem.getData("WordRecognized")
            # Typical format: [word, confidence, word2, conf2, ...]
            if isinstance(data, list) and len(data) >= 2:
                word = data[0]
                conf = data[1]
                if word and conf >= 0.35:
                    # debounce: avoid repeating same word many times
                    now = time.time()
                    if word == last_text and (now - last_time) < 2.0:
                        time.sleep(0.1)
                        continue
                    last_text = word
                    last_time = now

                    # stop
                    if word.lower() in ["στοπ", "stop"]:
                        tts.say("Εντάξει, σταματάω.")
                        break

                    # Ask Ollama (Ollama runs on laptop: use laptop IP reachable from NAO!)
                    reply = ollama_generate(
                        args.ollama_url,
                        args.ollama_model,
                        system,
                        word,
                        timeout=60,
                    )
                    if reply:
                        # Python2 naoqi prefers utf-8 bytes
                        tts.say(reply.encode("utf-8"))
            time.sleep(0.1)
    finally:
        try:
            asr.unsubscribe(subscriber)
        except Exception:
            pass

if __name__ == "__main__":
    main()