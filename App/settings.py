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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class AppSettings:
    rag_model: str
    rag_top_k: int
    rag_preload_retrieval_on_boot: bool
    rag_debug_chunks: bool
    rag_fallback_reply: str
    stt_whisper_model: str
    stt_language: str
    vosk_model_path: str
    push_max_seconds: int
    stt_debug_log_enabled: bool
    stt_debug_log_path: str
    continuous_vad_frame_ms: int
    continuous_vad_preroll_ms: int
    continuous_vad_min_speech_ms: int
    continuous_vad_silence_ms: int
    continuous_vad_max_segment_seconds: int
    continuous_vad_start_trigger_frames: int
    continuous_vad_start_threshold_db: int
    continuous_vad_stop_threshold_db: int
    continuous_vad_calibration_seconds: int
    continuous_speaking_tail_ms: int
    stt_normalization_enabled: bool
    stt_normalization_debug: bool
    stt_norm_enable_entity_injections: bool
    stt_norm_enable_gpt_normalization: bool
    stt_norm_entity_lexicon_path: str
    stt_normalization_use_gpt_fallback: bool
    stt_normalization_gpt_model: str
    stt_normalization_low_confidence: float
    stt_normalization_entity_threshold: int
    stt_normalization_max_ngram: int
    stt_normalization_gpt_timeout_sec: int


def build_settings() -> AppSettings:
    stt_lexicon_default = str((PROJECT_ROOT / "App" / "stt_entity_lexicon.json"))
    stt_norm_enable_gpt = _env_bool("STT_NORM_ENABLE_GPT_NORMALIZATION", True)
    if "STT_NORM_USE_GPT_FALLBACK" in os.environ:
        stt_norm_enable_gpt = _env_bool("STT_NORM_USE_GPT_FALLBACK", stt_norm_enable_gpt)

    return AppSettings(
        rag_model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        rag_top_k=_env_int("RAG_TOP_K", 10),
        rag_preload_retrieval_on_boot=_env_bool("RAG_PRELOAD_RETRIEVAL_ON_BOOT", True),
        rag_debug_chunks=_env_bool("RAG_DEBUG_CHUNKS", False),
        rag_fallback_reply="Δεν βρήκα αξιόπιστη απάντηση τώρα",
        stt_whisper_model=os.environ.get("STT_WHISPER_MODEL", "small"),
        stt_language=os.environ.get("STT_LANGUAGE", "el"),
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
        continuous_vad_frame_ms=_env_int("CONT_VAD_FRAME_MS", 30),
        continuous_vad_preroll_ms=_env_int("CONT_VAD_PREROLL_MS", 300),
        continuous_vad_min_speech_ms=_env_int("CONT_VAD_MIN_SPEECH_MS", 450),
        continuous_vad_silence_ms=_env_int("CONT_VAD_SILENCE_MS", 900),
        continuous_vad_max_segment_seconds=_env_int("CONT_VAD_MAX_SEGMENT_SEC", 12),
        continuous_vad_start_trigger_frames=_env_int("CONT_VAD_START_TRIGGER_FRAMES", 3),
        continuous_vad_start_threshold_db=_env_int("CONT_VAD_START_THRESHOLD_DB", 12),
        continuous_vad_stop_threshold_db=_env_int("CONT_VAD_STOP_THRESHOLD_DB", 6),
        continuous_vad_calibration_seconds=_env_int("CONT_VAD_CALIBRATION_SEC", 1),
        continuous_speaking_tail_ms=_env_int("CONT_SPEAKING_TAIL_MS", 400),
        stt_normalization_enabled=_env_bool("STT_NORMALIZATION_ENABLED", True),
        stt_normalization_debug=_env_bool("STT_NORMALIZATION_DEBUG", False),
        stt_norm_enable_entity_injections=_env_bool("STT_NORM_ENABLE_ENTITY_INJECTIONS", True),
        stt_norm_enable_gpt_normalization=stt_norm_enable_gpt,
        stt_norm_entity_lexicon_path=os.environ.get("STT_NORM_ENTITY_LEXICON_PATH", stt_lexicon_default),
        stt_normalization_use_gpt_fallback=stt_norm_enable_gpt,
        stt_normalization_gpt_model=os.environ.get("STT_NORM_GPT_MODEL", "gpt-4o-mini"),
        stt_normalization_low_confidence=_env_float("STT_NORM_LOW_CONFIDENCE", 0.82),
        stt_normalization_entity_threshold=_env_int("STT_NORM_ENTITY_THRESHOLD", 86),
        stt_normalization_max_ngram=_env_int("STT_NORM_MAX_NGRAM", 3),
        stt_normalization_gpt_timeout_sec=_env_int("STT_NORM_GPT_TIMEOUT_SEC", 20),
    )
