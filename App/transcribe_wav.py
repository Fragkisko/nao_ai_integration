#!/usr/bin/env python3
"""Simple helper: transcribe a WAV file with VOSK. Usage:
python transcribe_wav.py --file Tests/nao_last.wav --model "../vosk-model-el-gr-0.7"
"""
import argparse
import wave
import json
import sys
from pathlib import Path

from speech_to_text import load_custom_vocabulary_from_env

try:
    import vosk
except Exception as e:
    print("VOSK is not installed or cannot be imported:", e)
    sys.exit(2)


def transcribe(model_path: Path, wav_path: Path, use_vocab: bool = True) -> str:
    if not model_path.exists():
        raise RuntimeError(f"VOSK model not found at {model_path}")
    if not wav_path.exists():
        raise RuntimeError(f"WAV file not found at {wav_path}")

    model = vosk.Model(str(model_path))
    wf = wave.open(str(wav_path), 'rb')

    if use_vocab:
        vocab = load_custom_vocabulary_from_env()
    else:
        vocab = []

    if vocab:
        grammar = json.dumps([*vocab, "[unk]"], ensure_ascii=False)
        rec = vosk.KaldiRecognizer(model, wf.getframerate(), grammar)
        print(f"[INFO] STT vocabulary injection active ({len(vocab)} entries)")
    else:
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', '-f', required=True, help='Path to WAV file to transcribe')
    p.add_argument('--model', '-m', default=str(Path(__file__).resolve().parents[1] / 'vosk-model-el-gr-0.7'), help='Path to VOSK model directory')
    p.add_argument('--no-vocab', action='store_true', help='Disable vocabulary injection from STT_VOCAB_* env vars')
    args = p.parse_args()

    wav = Path(args.file)
    model = Path(args.model)
    try:
        text = transcribe(model, wav, use_vocab=not args.no_vocab)
        if text:
            print('Transcription:')
            print(text)
        else:
            print('[INFO] No text detected in file (empty transcription).')
    except Exception as e:
        print('[ERROR]', e)
        sys.exit(1)
