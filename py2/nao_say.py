# -*- coding: utf-8 -*-
import argparse
import io
from naoqi import ALProxy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True)
    p.add_argument("--port", type=int, default=9559)
    p.add_argument("--textfile", required=True)  # path to UTF-8 text file
    args = p.parse_args()

    # Read UTF-8 text safely in Python 2 as unicode
    with io.open(args.textfile, "r", encoding="utf-8") as f:
        text_u = f.read().strip()   # unicode

    # Convert to UTF-8 bytes (Python2 str) for naoqi bridge compatibility
    if isinstance(text_u, unicode):
        text = text_u.encode("utf-8")
    else:
        text = text_u

    tts = ALProxy("ALTextToSpeech", args.ip, args.port)

    # Optional: try to set Greek if supported
    try:
        tts.setLanguage("Greek")
    except Exception:
        pass

    tts.say(text)

if __name__ == "__main__":
    main()