import importlib.util
import os
import argparse
from pathlib import Path
import subprocess

# Parse CLI args early so we can override env vars before importing nao_common
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nao-ip", help="Override NAO IP address")
parser.add_argument("--nao-port", type=int, help="Override NAO port")
args, _ = parser.parse_known_args()
if args.nao_ip:
    os.environ["NAO_IP"] = args.nao_ip
if args.nao_port:
    os.environ["NAO_PORT"] = str(args.nao_port)

from nao_common import (
    NAO_IP,
    NAO_PORT,
    nao_say,
    tts_clip,
)

# -----------------------------
# Config (all overridable by env)
# -----------------------------
RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
RAG_DEBUG_CHUNKS = os.environ.get("RAG_DEBUG_CHUNKS", "0").strip().lower() in {"1", "true", "yes", "on"}
RAG_FALLBACK_REPLY = "Δεν βρήκα αξιόπιστη απάντηση τώρα"


def _load_rag_cli_module():
    rag_cli_path = Path(__file__).resolve().parents[1] / "Rag-Chabot" / "Generator" / "rag_cli.py"
    spec = importlib.util.spec_from_file_location("rag_cli_runtime", str(rag_cli_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rag_cli module from {rag_cli_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    rag_module = None
    rag_engine = None
    try:
        rag_module = _load_rag_cli_module()
        rag_engine = rag_module.RAGEngine(
            out="../Embedding/out",
            model=RAG_MODEL,
            quiet=True,
            k=RAG_TOP_K,
            structured_mode="llm",
        )
    except Exception as e:
        print(f"[WARN] RAG init error: {e}")

    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"RAG model={RAG_MODEL}")
    print(f"RAG debug chunks={RAG_DEBUG_CHUNKS}")
    print("Γράψε κάτι (Enter για έξοδο):")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            break

        rag_result = {"answer": None, "error": "RAG engine unavailable", "chunks": []}
        if rag_engine is not None:
            try:
                rag_result = rag_engine.answer_query(user_text)
            except Exception as e:
                rag_result = {"answer": None, "error": f"RAG runtime error: {e}", "chunks": []}

        if RAG_DEBUG_CHUNKS and rag_module is not None:
            for i, item in enumerate(rag_result.get("chunks", []), start=1):
                print(rag_module.format_chunk(item, i))

        reply = (rag_result.get("answer") or "").strip()
        if not reply:
            if rag_result.get("error"):
                print(f"[WARN] {rag_result['error']}")
            reply = RAG_FALLBACK_REPLY

        print("NAO:", reply)

        try:
            nao_say(tts_clip(reply))
        except subprocess.TimeoutExpired:
            print("[WARN] NAO timeout")
        except Exception as e:
            print(f"[WARN] NAO error: {e}")

    print("Τέλος")