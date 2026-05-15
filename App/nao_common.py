import json
import os
import subprocess
import tempfile
import urllib.request

PY2_EXE = os.environ.get("PY2_EXE", r"C:\Python27\python.exe")
PYNAOQI_LIB = os.environ.get(
    "PYNAOQI_LIB",
    r"C:\NaoQi\pynaoqi-python2.7-2.8.6.23-win64-vs2015-20191127_152649\lib",
)

NAO_IP = os.environ.get("NAO_IP", "127.0.0.1")
NAO_PORT = int(os.environ.get("NAO_PORT", "9559"))
NAO_TTS_TIMEOUT_SEC = int(os.environ.get("NAO_TTS_TIMEOUT_SEC", "30"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
MAX_TTS_LEN = int(os.environ.get("MAX_TTS_LEN", "250"))

DEFAULT_TEXT_SYSTEM_PROMPT = (
    "Είσαι το ρομπότ NAO και μιλάς σε επισκέπτες της σχολής ΔΙΠΑΕ. "
    "Μίλα ΜΟΝΟ ελληνικά, απλά και φυσικά. "
    "Απάντα πολύ σύντομα (μέχρι 1 πρόταση). "
    "Μην χρησιμοποιείς αγγλικά ή περίεργες λέξεις. "
    "Αν δεν κατάλαβες, πες: Μπορείς να το πεις αλλιώς;"
)

DEFAULT_VOICE_SYSTEM_PROMPT = (
    "Μίλα ΜΟΝΟ με σωστά Νέα Ελληνικά, με απλές και φυσικές λέξεις. "
    "Απάντα σύντομα, σε μία πρόταση. "
    "Μην χρησιμοποιείς αγγλικά, greeklish ή δύσκολες εκφράσεις. "
    "Αν δεν κατάλαβες, πες: Μπορείς να το πεις αλλιώς;"
)


def get_system_prompt(default_prompt: str) -> str:
    return os.environ.get("OLLAMA_SYSTEM", default_prompt)


def get_ollama_timeout(default_sec: int) -> int:
    return int(os.environ.get("OLLAMA_TIMEOUT_SEC", str(default_sec)))


def get_ollama_temperature(default: float = 0.2) -> float:
    return float(os.environ.get("OLLAMA_TEMPERATURE", str(default)))


def build_ollama_options(temperature: float) -> dict:
    return {
        "temperature": temperature,
        "top_p": 0.8,
        "repeat_penalty": 1.1,
    }


def clean_response(text: str) -> str:
    t = (text or "").strip()

    if len(t) < 2:
        return "Μπορείς να το πεις αλλιώς;"

    if any(word in t.lower() for word in ["the", "is", "and"]):
        return "Μπορείς να το πεις αλλιώς;"

    return t


def rule_based_response(text: str) -> str | None:
    t = text.lower()

    if "γεια" in t:
        return "Γεια σου! Καλώς ήρθες!"
    if "τι κάνεις" in t:
        return "Είμαι καλά, ευχαριστώ!"
    if "όνομα" in t:
        return "Με λένε NAO."
    if "καλά" in t:
        return "Χαίρομαι που είσαι καλά!"

    return None


def tts_clip(text: str, max_len: int = MAX_TTS_LEN) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_len:
        return t

    cut = t[:max_len]
    for sep in [".", "!", "?", "·", ";", ","]:
        i = cut.rfind(sep)
        if i > max_len // 3:
            return cut[: i + 1]
    return cut


def nao_say(
    text: str,
    *,
    nao_ip: str = NAO_IP,
    nao_port: int = NAO_PORT,
    py2_exe: str = PY2_EXE,
    pynaoqi_lib: str = PYNAOQI_LIB,
    timeout_sec: int = NAO_TTS_TIMEOUT_SEC,
):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(app_dir)
    script = os.path.join(repo_root, "py2", "nao_say.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = pynaoqi_lib

    fd, path = tempfile.mkstemp(prefix="nao_tts_", suffix=".txt")
    os.close(fd)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        cmd = [py2_exe, script, "--ip", nao_ip, "--port", str(nao_port), "--textfile", path]
        subprocess.run(cmd, env=env, check=True, timeout=timeout_sec)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def ollama_generate(
    prompt: str,
    *,
    system_prompt: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    timeout_sec: int = 120,
    options: dict | None = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    return (out.get("response") or "").strip()
