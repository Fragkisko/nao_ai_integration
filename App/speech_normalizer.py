import os
import re
import unicodedata
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    _rapidfuzz = importlib.import_module("rapidfuzz")
    fuzz = getattr(_rapidfuzz, "fuzz", None)
    process = getattr(_rapidfuzz, "process", None)

    RAPIDFUZZ_AVAILABLE = bool(fuzz is not None and process is not None)
except Exception:
    fuzz = None
    process = None
    RAPIDFUZZ_AVAILABLE = False


DEFAULT_ENTITY_DICTIONARY: Dict[str, List[str]] = {
    "ΔΙΠΑΕ": [
        "διπαε",
        "διπα ε",
        "διπα έκανε",
        "διπα κανε",
        "zipaae",
        "zipae",
        "ζιπαε",
        "ζυπαε",
    ],
    "ΠΜΣ": [
        "πμσ",
        "που μου σου",
        "που μου σου εφεί",
        "μεταπτυχιακο",
        "μεταπτυχιακό",
    ],
    "Σίνδος": ["σινδος", "σίνδος", "συνδος", "συντος"],
    "Γραμματεία": ["γραμματεια", "γραματεια", "γραμματeια"],
    "Erasmus": ["erasmus", "ερασμους", "ερασμουσ", "εράσμους"],
    "Βιβλιοθήκη": ["βιβλιοθηκη", "βιβλιοθήκη"],
    "Αλεξάνδρεια Πανεπιστημιούπολη": [
        "αλεξανδρεια πανεπιστημιουπολη",
        "αλεξάνδρεια πανεπιστημιούπολη",
        "πανεπιστημιουπολη αλεξανδρειας",
    ],
    "Τμήμα Τεχνολογίας Διαδικτύου": [
        "τμημα τεχνολογιας διαδικτυου",
        "τμήμα τεχνολογίας διαδικτύου",
        "στεχνολογια της διαδικτυου",
        "τεχνολογια διαδικτυου",
    ],
}

SYSTEM_PROMPT = (
    "You are a Greek speech-to-text correction assistant. "
    "Your task is NOT to answer the question. "
    "Your task is ONLY to correct transcription errors produced by a speech recognition system.\n\n"
    "Rules:\n"
    "- Preserve the original meaning.\n"
    "- Correct only probable speech recognition mistakes.\n"
    "- Keep the sentence in Greek.\n"
    "- Do not add information that the user did not say.\n"
    "- Do not explain your changes.\n"
    "- Return ONLY the corrected sentence.\n"
    "- If you are uncertain, perform the minimum necessary corrections.\n\n"
    "The application is a voice assistant for the International Hellenic University (ΔΙΠΑΕ). "
    "Common entities include ΔΙΠΑΕ, Σίνδος, Αλεξάνδρεια Πανεπιστημιούπολη, ΠΜΣ, Γραμματεία, "
    "Erasmus, Βιβλιοθήκη, and terms related to Τμήμα Τεχνολογίας Διαδικτύου.\n\n"
    "Examples:\n"
    "Input: \"Που βρίσκεται το ZIPAAE\"\n"
    "Output: \"Πού βρίσκεται το ΔΙΠΑΕ;\"\n"
    "Input: \"Που μου σου εφεί η στεχνολογία της διαδικτύου\"\n"
    "Output: \"ΠΜΣ στην Τεχνολογία Διαδικτύου\"\n"
    "Input: \"Πως μπορω να μετακινηθω προς το διπα εκανε\"\n"
    "Output: \"Πώς μπορώ να μετακινηθώ προς το ΔΙΠΑΕ;\""
)

_WORD_RE = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-ώ0-9]+", flags=re.UNICODE)
_DUP_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpeechNormalizationConfig:
    enabled: bool = True
    enable_entity_injections: bool = True
    use_gpt_fallback: bool = True
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    entity_lexicon_path: str = ""
    deterministic_match_threshold: int = 86
    low_confidence_threshold: float = 0.82
    max_ngram: int = 3
    timeout_sec: int = 20
    debug: bool = False


@dataclass(frozen=True)
class SpeechNormalizationResult:
    original_text: str
    deterministic_text: str
    corrected_text: str
    source: str
    confidence: float
    matched_entities: Tuple[str, ...]
    used_gpt: bool
    gpt_error: str = ""

    @property
    def changed(self) -> bool:
        return (self.corrected_text or "").strip() != (self.original_text or "").strip()


class SpeechNormalizer:
    def __init__(
        self,
        config: Optional[SpeechNormalizationConfig] = None,
        entity_dictionary: Optional[Dict[str, Sequence[str]]] = None,
    ):
        self.config = config or SpeechNormalizationConfig()
        self.entity_dictionary = self._resolve_entity_dictionary(entity_dictionary)
        self._alias_index = self._build_alias_index(self.entity_dictionary)

    def _resolve_entity_dictionary(
        self,
        entity_dictionary: Optional[Dict[str, Sequence[str]]],
    ) -> Dict[str, List[str]]:
        if entity_dictionary is not None:
            return _normalize_entity_dictionary(entity_dictionary)

        merged: Dict[str, List[str]] = _normalize_entity_dictionary(DEFAULT_ENTITY_DICTIONARY)
        external = load_entity_lexicon(self.config.entity_lexicon_path)
        if external:
            for canonical, aliases in external.items():
                if canonical not in merged:
                    merged[canonical] = []
                merged[canonical].extend(aliases)
                merged[canonical] = sorted(set(alias for alias in merged[canonical] if alias.strip()))
        return merged

    @staticmethod
    def _build_alias_index(entity_dictionary: Dict[str, Sequence[str]]) -> List[Tuple[str, str]]:
        alias_index: List[Tuple[str, str]] = []
        for canonical, aliases in entity_dictionary.items():
            canonical_folded = _fold_for_match(canonical)
            alias_index.append((canonical, canonical_folded))
            for alias in aliases:
                alias_index.append((canonical, _fold_for_match(alias)))
        return alias_index

    def normalize(self, text: str) -> SpeechNormalizationResult:
        original = (text or "").strip()
        if not original:
            return SpeechNormalizationResult(
                original_text="",
                deterministic_text="",
                corrected_text="",
                source="none",
                confidence=0.0,
                matched_entities=(),
                used_gpt=False,
            )

        if not self.config.enabled:
            return SpeechNormalizationResult(
                original_text=original,
                deterministic_text=original,
                corrected_text=original,
                source="disabled",
                confidence=1.0,
                matched_entities=(),
                used_gpt=False,
            )

        deterministic_input = _basic_text_normalization(original)
        if self.config.enable_entity_injections:
            deterministic_text, confidence, entities = self._apply_entity_corrections(deterministic_input)
        else:
            deterministic_text, confidence, entities = deterministic_input, 0.0, []

        should_use_gpt = (
            self.config.use_gpt_fallback
            and (
                len(entities) == 0
                or confidence < float(self.config.low_confidence_threshold)
            )
        )

        if should_use_gpt:
            gpt_text, gpt_error = self._rewrite_with_gpt(deterministic_text)
            if gpt_text:
                return SpeechNormalizationResult(
                    original_text=original,
                    deterministic_text=deterministic_text,
                    corrected_text=gpt_text,
                    source="gpt_fallback",
                    confidence=confidence,
                    matched_entities=tuple(entities),
                    used_gpt=True,
                    gpt_error=gpt_error,
                )

            return SpeechNormalizationResult(
                original_text=original,
                deterministic_text=deterministic_text,
                corrected_text=deterministic_text,
                source="deterministic_only",
                confidence=confidence,
                matched_entities=tuple(entities),
                used_gpt=True,
                gpt_error=gpt_error,
            )

        return SpeechNormalizationResult(
            original_text=original,
            deterministic_text=deterministic_text,
            corrected_text=deterministic_text,
            source="deterministic_only",
            confidence=confidence,
            matched_entities=tuple(entities),
            used_gpt=False,
        )

    def _apply_entity_corrections(self, text: str) -> Tuple[str, float, List[str]]:
        spans = list(_WORD_RE.finditer(text))
        if not spans:
            return text, 0.0, []

        words = [m.group(0) for m in spans]
        replacements: List[Tuple[int, int, str, str, float]] = []
        max_ngram = max(1, int(self.config.max_ngram))

        for i in range(len(words)):
            for n in range(max_ngram, 0, -1):
                if i + n > len(words):
                    continue
                candidate = " ".join(words[i : i + n])
                best_entity, best_score = self._best_entity_match(candidate)
                if not best_entity:
                    continue
                if best_score < float(self.config.deterministic_match_threshold):
                    continue

                start = spans[i].start()
                end = spans[i + n - 1].end()
                replacements.append((start, end, best_entity, candidate, best_score))
                break

        if not replacements:
            return text, 0.0, []

        # Keep highest-confidence, non-overlapping replacements.
        replacements.sort(key=lambda item: (item[4], (item[1] - item[0])), reverse=True)
        accepted: List[Tuple[int, int, str, str, float]] = []
        for replacement in replacements:
            start, end = replacement[0], replacement[1]
            overlaps = any(not (end <= a_start or start >= a_end) for a_start, a_end, *_ in accepted)
            if not overlaps:
                accepted.append(replacement)

        accepted.sort(key=lambda item: item[0])

        out_parts: List[str] = []
        cursor = 0
        matched_entities: List[str] = []
        scores: List[float] = []
        for start, end, canonical, _candidate, score in accepted:
            if cursor < start:
                out_parts.append(text[cursor:start])
            out_parts.append(canonical)
            cursor = end
            matched_entities.append(canonical)
            scores.append(score)
        if cursor < len(text):
            out_parts.append(text[cursor:])

        corrected = "".join(out_parts)
        confidence = sum(scores) / (100.0 * len(scores)) if scores else 0.0
        return corrected, confidence, sorted(set(matched_entities))

    def _best_entity_match(self, candidate: str) -> Tuple[Optional[str], float]:
        folded_candidate = _fold_for_match(candidate)
        if not folded_candidate:
            return None, 0.0

        if RAPIDFUZZ_AVAILABLE and process is not None and fuzz is not None:
            alias_values = [alias for _, alias in self._alias_index]
            best = process.extractOne(
                folded_candidate,
                alias_values,
                scorer=fuzz.WRatio,
                score_cutoff=0,
            )
            if best is None:
                return None, 0.0
            alias_match = str(best[0])
            score = float(best[1])
            for canonical, alias in self._alias_index:
                if alias == alias_match:
                    return canonical, score
            return None, 0.0

        best_entity: Optional[str] = None
        best_score = 0.0
        for canonical, alias in self._alias_index:
            score = _simple_similarity(folded_candidate, alias) * 100.0
            if score > best_score:
                best_score = score
                best_entity = canonical
        return best_entity, best_score

    def _rewrite_with_gpt(self, text: str) -> Tuple[str, str]:
        api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return "", "OPENAI_API_KEY is not set"

        base_url = (self.config.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"

        payload = {
            "model": self.config.openai_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=max(5, int(self.config.timeout_sec)),
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            content = (content or "").strip()
            if not content:
                return "", "empty GPT correction"
            return content, ""
        except Exception as e:
            return "", str(e)


def _fold_for_match(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("ς", "σ")
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_accents = re.sub(r"[^0-9a-zα-ω\s]", " ", without_accents)
    return _DUP_SPACE_RE.sub(" ", without_accents).strip()


def _simple_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la = len(a)
    lb = len(b)
    common = 0
    for ch_a, ch_b in zip(a, b):
        if ch_a == ch_b:
            common += 1
    return common / float(max(la, lb))


def _basic_text_normalization(text: str) -> str:
    out = (text or "").strip()

    # Normalize punctuation variants and spacing.
    out = out.replace("?", ";")
    out = out.replace("·", ".")
    out = out.replace("…", "...")
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,.;:!?])(?!\s|$)", r"\1 ", out)
    out = _DUP_SPACE_RE.sub(" ", out)

    # Lowercase first and let deterministic entity replacement restore canonical casing.
    out = out.lower()

    # Keep final punctuation in a consistent Greek-friendly form.
    out = re.sub(r"[;]{2,}", ";", out)
    out = re.sub(r"[.]{4,}", "...", out)

    return out.strip()


def build_speech_normalizer_from_env(*, enabled: bool, debug: bool = False) -> SpeechNormalizer:
    enable_gpt_norm = _env_bool("STT_NORM_ENABLE_GPT_NORMALIZATION", True)
    # Backward compatibility with previous env key.
    if "STT_NORM_USE_GPT_FALLBACK" in os.environ:
        enable_gpt_norm = _env_bool("STT_NORM_USE_GPT_FALLBACK", enable_gpt_norm)

    lexicon_default = str(Path(__file__).resolve().with_name("stt_entity_lexicon.json"))

    cfg = SpeechNormalizationConfig(
        enabled=enabled,
        enable_entity_injections=_env_bool("STT_NORM_ENABLE_ENTITY_INJECTIONS", True),
        use_gpt_fallback=enable_gpt_norm,
        openai_model=os.environ.get("STT_NORM_GPT_MODEL", "gpt-4o-mini"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        entity_lexicon_path=os.environ.get("STT_NORM_ENTITY_LEXICON_PATH", lexicon_default),
        deterministic_match_threshold=_env_int("STT_NORM_ENTITY_THRESHOLD", 86),
        low_confidence_threshold=_env_float("STT_NORM_LOW_CONFIDENCE", 0.82),
        max_ngram=_env_int("STT_NORM_MAX_NGRAM", 3),
        timeout_sec=_env_int("STT_NORM_GPT_TIMEOUT_SEC", 20),
        debug=debug,
    )
    return SpeechNormalizer(config=cfg)


def load_entity_lexicon(path_value: str) -> Dict[str, List[str]]:
    path_str = (path_value or "").strip()
    if not path_str:
        return {}

    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return {}

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            import json

            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            return _parse_entity_lexicon_payload(payload)

        if suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except Exception:
                return {}
            with path.open("r", encoding="utf-8") as fp:
                payload = yaml.safe_load(fp)
            return _parse_entity_lexicon_payload(payload)
    except Exception:
        return {}

    return {}


def _parse_entity_lexicon_payload(payload: Any) -> Dict[str, List[str]]:
    if isinstance(payload, dict) and "entities" in payload and isinstance(payload.get("entities"), list):
        result: Dict[str, List[str]] = {}
        for item in payload.get("entities", []):
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("canonical", "")).strip()
            if not canonical:
                continue
            aliases_raw = item.get("aliases", [])
            aliases = [str(alias).strip() for alias in aliases_raw if str(alias).strip()]
            result[canonical] = sorted(set(aliases))
        return _normalize_entity_dictionary(result)

    if isinstance(payload, dict):
        return _normalize_entity_dictionary(payload)

    return {}


def _normalize_entity_dictionary(raw: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for canonical, aliases in raw.items():
        canonical_key = str(canonical).strip()
        if not canonical_key:
            continue

        if aliases is None:
            alias_list: List[str] = []
        elif isinstance(aliases, str):
            alias_list = [aliases.strip()] if aliases.strip() else []
        else:
            alias_list = [str(alias).strip() for alias in aliases if str(alias).strip()]

        normalized[canonical_key] = sorted(set(alias_list))
    return normalized


def _env_bool(name: str, default: bool) -> bool:
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
