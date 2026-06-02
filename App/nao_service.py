import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from speech_to_text import transcribe_file_with_whisper


def run_nao_record_and_transcribe(whisper_model, default_nao_ip: str, default_nao_port: int) -> str:
    """Run remote NAO recording and return transcribed text (or empty string)."""
    nao_user = os.environ.get("NAO_USER", "nao")
    nao_ip_runtime = os.environ.get("NAO_IP", default_nao_ip)
    _nao_port_runtime = os.environ.get("NAO_PORT", str(default_nao_port))
    record_duration = int(os.environ.get("NAO_RECORD_SEC", "5"))
    prep_seconds = int(os.environ.get("NAO_PREP_SEC", "3"))

    print(f"Prepare: you will have {prep_seconds}s to get ready, then NAO will record for {record_duration}s.")
    try:
        for i in range(prep_seconds, 0, -1):
            print(f"Speak in: {i}...", end="\r")
            time.sleep(1)
        print("Go! Recording starting...".ljust(60))
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled prep.")
        return ""

    remote_py = "/home/nao/nao_record.py"
    remote_out = "/home/nao/record.wav"

    ssh_cmd = [
        "ssh",
        f"{nao_user}@{nao_ip_runtime}",
        "/usr/bin/python2 {} --ip {} --duration {} --out {}".format(
            remote_py,
            nao_ip_runtime,
            record_duration,
            remote_out,
        ),
    ]

    print(f"Starting remote recording on NAO {nao_ip_runtime} for {record_duration}s...")
    try:
        subprocess.check_call(ssh_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Remote record failed: {e}")
        print("Tip: ensure /home/nao/nao_record.py exists on the robot and SSH password/key available.")
        return ""

    tmp_local = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_local_path = tmp_local.name
    tmp_local.close()

    scp_cmd = ["scp", f"{nao_user}@{nao_ip_runtime}:{remote_out}", tmp_local_path]
    try:
        subprocess.check_call(scp_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] SCP download failed: {e}")
        try:
            os.remove(tmp_local_path)
        except Exception:
            pass
        return ""

    try:
        user_text = transcribe_file_with_whisper(whisper_model, tmp_local_path)
    except Exception as e:
        print(f"[WARN] STT-from-NAO-file error: {e}")
        user_text = ""

    tests_dir = Path(__file__).resolve().parents[1] / "Tests"
    try:
        if not user_text:
            try:
                tests_dir.mkdir(parents=True, exist_ok=True)
                saved = tests_dir / "nao_last.wav"
                shutil.move(tmp_local_path, str(saved))
                print(f"[DEBUG] Saved NAO recording for inspection: {saved}")
            except Exception:
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
        try:
            os.remove(tmp_local_path)
        except Exception:
            pass

    if not user_text:
        print("[WARN] Δεν ανιχνεύτηκε κείμενο στο sample.")
    else:
        print("Καταγραφή (NAO):", user_text)

    return user_text
