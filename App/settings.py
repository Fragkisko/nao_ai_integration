import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_from_project_root() -> None:
    """Load .env from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
    except Exception:
        pass


def apply_cli_overrides(argv: Optional[Sequence[str]] = None):
    """Parse known CLI args and override env vars consumed by nao_common."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nao-ip", help="Override NAO IP address")
    parser.add_argument("--nao-port", type=int, help="Override NAO port")
    args, _ = parser.parse_known_args(argv)

    if args.nao_ip:
        os.environ["NAO_IP"] = args.nao_ip
    if args.nao_port:
        os.environ["NAO_PORT"] = str(args.nao_port)

    return args


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class AppSettings:
    rag_model: str
    rag_top_k: int
    rag_preload_retrieval_on_boot: bool
    rag_debug_chunks: bool
    rag_fallback_reply: str
    vosk_model_path: str
    push_max_seconds: int
    stt_debug_log_enabled: bool
    stt_debug_log_path: str


def build_settings() -> AppSettings:
    return AppSettings(
        rag_model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        rag_top_k=_env_int("RAG_TOP_K", 5),
        rag_preload_retrieval_on_boot=_env_bool("RAG_PRELOAD_RETRIEVAL_ON_BOOT", True),
        rag_debug_chunks=_env_bool("RAG_DEBUG_CHUNKS", False),
        rag_fallback_reply="Δεν βρήκα αξιόπιστη απάντηση τώρα",
        vosk_model_path=os.environ.get(
            "VOSK_MODEL_PATH",
            str(PROJECT_ROOT / "models" / "vosk-model-gr"),
        ),
        push_max_seconds=_env_int("PUSH_MAX_SECONDS", 15),
        stt_debug_log_enabled=_env_bool("STT_DEBUG_LOG_ENABLED", True),
        stt_debug_log_path=os.environ.get(
            "STT_DEBUG_LOG_PATH",
            str(PROJECT_ROOT / "Tests" / "stt_debug_log.csv"),
        ),
    )
