# -*- coding: utf-8 -*-
import argparse
import time
import io
import subprocess
import math
import os

try:
    from naoqi import ALProxy
    _have_naoqi = True
except Exception:
    _have_naoqi = False


def record_with_naoqi(ip, port, out, duration):
    rec = ALProxy("ALAudioRecorder", ip, port)
    rec.startMicrophonesRecording(out, "wav", 16000)
    print("Recording...", duration, "seconds (naoqi)")
    time.sleep(float(duration))
    rec.stopMicrophonesRecording()
    print("Saved:", out)


def record_with_arecord(out, duration):
    dur = int(math.ceil(float(duration)))
    cmd = ['arecord', '-r', '16000', '-f', 'S16_LE', '-c', '1', '-d', str(dur), out]
    print("Recording with arecord:", ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print("Saved:", out)
    except Exception as e:
        print("[ERROR] arecord failed:", e)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True, help="NAO IP")
    p.add_argument("--port", type=int, default=9559, help="NAO port")
    p.add_argument("--duration", type=float, default=5.0, help="Seconds to record")
    p.add_argument("--out", default="/home/nao/record.wav", help="Output path on robot")
    args = p.parse_args()

    # Try naoqi first if available, otherwise fall back to arecord
    if _have_naoqi:
        try:
            record_with_naoqi(args.ip, args.port, args.out, args.duration)
            return
        except Exception as e:
            print("[WARN] naoqi recording failed, falling back to arecord:", e)

    try:
        record_with_arecord(args.out, args.duration)
    except Exception as e:
        print("[ERROR] Recording failed with both naoqi and arecord:", e)


if __name__ == '__main__':
    main()
