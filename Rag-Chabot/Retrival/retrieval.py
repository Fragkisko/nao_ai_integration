"""Simple retrieval layer for RAG system.

Includes:
- Imports & constants
- Loading layer (model, faiss/index, embeddings, metadata)
- Query processing (normalize + build E5 query)
- Search function (FAISS or numpy fallback)

Run as script with `--query` for a quick check.
"""
from pathlib import Path
import json
import logging
import unicodedata
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from difflib import SequenceMatcher

from bm25_index import BM25Index, load_or_build_bm25_index
from hybrid_fusion import reciprocal_rank_fusion

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ----------------------
# Config / Constants
# ----------------------
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 5
SCORE_THRESHOLD = 0.2
MAX_CONTEXT_CHARS = 12000

EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
FAISS_INDEX_FILENAME = "faiss.index"
BM25_INDEX_FILENAME = "bm25_index.json"

HYBRID_DENSE_TOP_N = 20
HYBRID_SPARSE_TOP_N = 20
RRF_K = 60
PROGRAM_MATCH_BONUS = 5.0

CATEGORY_KEYWORD_BONUS = 10.0
NO_EVIDENCE_ANSWER = "Δεν βρέθηκε σχετική πληροφορία."

PROGRAM_CANONICAL = {
    "intelligent_internet_technologies": "Ευφυείς Τεχνολογίες Διαδικτύου",
    "new_tech_innovation_didactics": "Νέες Τεχνολογίες, Καινοτομία και Διδακτική",
    "digital_soft_skills": "Ψηφιακές και Ήπιες Δεξιότητες",
}

PROGRAM_DETECTION_RULES = {
    PROGRAM_CANONICAL["intelligent_internet_technologies"]: ["ευφυ", "διαδικτυ"],
    PROGRAM_CANONICAL["new_tech_innovation_didactics"]: ["καινοτομι", "διδακτικ"],
    PROGRAM_CANONICAL["digital_soft_skills"]: ["ψηφιακ", "ηπιες δεξιοτητες", "ήπιες δεξιότητες"],
}

PROGRAM_SOURCE_PATTERNS = {
    PROGRAM_CANONICAL["intelligent_internet_technologies"]: [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Ευφυείς-Τεχνολογίες-Διαδικτύου",
    ],
    PROGRAM_CANONICAL["new_tech_innovation_didactics"]: [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Νέες τεχνολογίες-καινοτομία-και-διδακτική",
    ],
    PROGRAM_CANONICAL["digital_soft_skills"]: [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Ψηφιακές-και-ήπιες-δεξιότητες-στις-επιστήμες-της-αγωγής",
    ],
}

CATEGORY_KEYWORDS = {
    "fees": ["διδακτρα", "δίδακτρα", "κοστος", "κόστος", "τελη", "τέλη", "ευρω", "ευρώ"],
    "duration": ["διαρκεια", "διάρκεια", "εξαμηνα", "εξάμηνα", "χρονια", "χρόνια", "ετη", "έτη"],
    "admissions": ["εισακτεοι", "εισακτέοι", "θεσεις", "θέσεις", "αιτησεις", "αιτήσεις"],
    "curriculum": ["μαθηματα", "μαθήματα", "ects"],
}

CATEGORY_BONUS_TERMS = {
    "fees": ["δίδακτρα", "διδακτρα", "τέλη φοίτησης", "τελη φοιτησης", "ευρώ", "ευρω", "500"],
    "duration": ["εξάμηνα", "εξαμηνα", "διάρκεια", "διαρκεια", "έτη", "ετη"],
    "admissions": ["εισακτέοι", "εισακτεοι", "θέσεις", "θεσεις", "φοιτητές", "φοιτητες"],
    "curriculum": ["ects", "μαθήματα", "μαθηματα"],
}

POSTGRAD_QUERY_HINTS = (
    "μεταπτυχ",
    "πμσ",
    "pms",
    "master",
    "msc",
)

POSTGRAD_PROGRAM_SOURCES = {
    "digital_soft_skills": [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Ψηφιακές-και-ήπιες-δεξιότητες-στις-επιστήμες-της-αγωγής",
    ],
    "new_tech_innovation_didactics": [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Νέες τεχνολογίες-καινοτομία-και-διδακτική",
    ],
    "intelligent_internet_technologies": [
        "Οδηγός-Σπουδών-Μεταπτυχιακο-Ευφυείς-Τεχνολογίες-Διαδικτύου",
    ],
}

POSTGRAD_GENERAL_SOURCES = ["Μεταπτυχιακών_Σπουδών"]

POSTGRAD_PROGRAM_QUERY_ALIASES = {
    "digital_soft_skills": [
        "ψηφιακες και ηπιες δεξιοτητες στις επιστημες της αγωγης",
        "ψηφιακες ηπιες δεξιοτητες",
        "επιστημες της αγωγης",
    ],
    "new_tech_innovation_didactics": [
        "νεες τεχνολογιες καινοτομια και διδακτικη",
        "νεες τεχνολογιες καινοτομια",
        "διδακτικη",
    ],
    "intelligent_internet_technologies": [
        "ευφυεις τεχνολογιες διαδικτυου",
        "ευφυεις τεχνολογιες",
    ],
}

ALL_POSTGRAD_GUIDE_SOURCES = [
    src for sources in POSTGRAD_PROGRAM_SOURCES.values() for src in sources
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------
# Loading Layer
# ----------------------
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def load_faiss_index(index_path: Path) -> Optional[faiss.Index]:
    if not _HAS_FAISS:
        logger.info("faiss not available, skipping index load")
        return None
    if not index_path.exists():
        logger.info("faiss index not found at %s", index_path)
        return None
    logger.info("Loading FAISS index from %s", index_path)
    return faiss.read_index(str(index_path))


def load_embeddings_np(emb_path: Path) -> Optional[np.ndarray]:
    if not emb_path.exists():
        logger.info("embeddings file not found at %s", emb_path)
        return None
    logger.info("Loading embeddings array from %s", emb_path)
    return np.load(str(emb_path))


def load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    logger.info("Loading metadata from %s", meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------
# Query Processing Layer
# ----------------------
def normalize_query(q: str) -> str:
    # lower, strip, unicode normalize, collapse spaces
    if q is None:
        return ""
    q = q.strip()
    q = unicodedata.normalize("NFKC", q)
    q = q.lower()
    q = re.sub(r"\s+", " ", q)
    return q


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-zα-ω]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _source_from_item(item: Dict[str, Any]) -> str:
    meta = item.get("meta") or {}
    return str(item.get("source") or meta.get("source") or meta.get("filename") or "")


def _source_from_meta(meta: Dict[str, Any]) -> str:
    return str(meta.get("source") or meta.get("filename") or "")


def _contains_any(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    key = _normalize_for_match(text)
    for pat in patterns:
        p_key = _normalize_for_match(pat)
        if p_key and p_key in key:
            return True
    return False


def _canonical_program_label(raw_program: str) -> Optional[str]:
    key = _normalize_for_match(raw_program)
    for canonical, hints in PROGRAM_DETECTION_RULES.items():
        if _contains_any(key, hints):
            return canonical
    return None


def detect_program(query: str) -> Optional[str]:
    q_key = _normalize_for_match(query)
    for canonical, hints in PROGRAM_DETECTION_RULES.items():
        if _contains_any(q_key, hints):
            return canonical
    return None


def detect_question_category(query: str) -> str:
    q_key = _normalize_for_match(query)
    for category in ("fees", "duration", "admissions", "curriculum"):
        if _contains_any(q_key, CATEGORY_KEYWORDS[category]):
            return category
    return "general"


def analyze_query(query: str) -> Dict[str, Optional[str]]:
    return {
        "program": detect_program(query),
        "category": detect_question_category(query),
    }


def infer_program_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    explicit = meta.get("program")
    if explicit:
        canonical = _canonical_program_label(str(explicit))
        if canonical:
            return canonical

    source = _source_from_meta(meta)
    for canonical, patterns in PROGRAM_SOURCE_PATTERNS.items():
        if _match_any_pattern(source, patterns):
            return canonical

    return None


def filter_metadata_by_program(metadata: List[Dict[str, Any]], program: Optional[str]) -> List[int]:
    if not program:
        return list(range(len(metadata)))

    allowed: List[int] = []
    for idx, meta in enumerate(metadata):
        if infer_program_from_meta(meta) == program:
            allowed.append(idx)
    return allowed


def _keyword_bonus_for_text(text: str, category: str) -> float:
    terms = CATEGORY_BONUS_TERMS.get(category, [])
    if not terms:
        return 0.0
    return CATEGORY_KEYWORD_BONUS if _contains_any(text, terms) else 0.0


def apply_bonus_rerank(results: List[Dict[str, Any]], category: str, detected_program: Optional[str]) -> List[Dict[str, Any]]:
    boosted: List[Dict[str, Any]] = []
    for item in results:
        meta = item.get("meta") or {}
        text = str(meta.get("retrieval_text") or meta.get("text") or "")

        keyword_bonus = _keyword_bonus_for_text(text, category)
        chunk_program = infer_program_from_meta(meta)
        program_bonus = PROGRAM_MATCH_BONUS if detected_program and chunk_program == detected_program else 0.0

        base_score = float(item.get("score", 0.0))
        final_score = base_score + keyword_bonus + program_bonus

        item["base_score"] = base_score
        item["keyword_bonus"] = keyword_bonus
        item["program_match_bonus"] = program_bonus
        item["score"] = final_score
        boosted.append(item)

    boosted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return boosted


def filtered_bm25_search(index: BM25Index,
                         query: str,
                         metadata: List[Dict[str, Any]],
                         top_n: int,
                         allowed_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    if allowed_indices is None:
        return index.search(query, top_n=top_n, metadata=metadata)

    if not allowed_indices:
        return []

    scores = index.score_query(query)
    ranked = sorted(
        allowed_indices,
        key=lambda i: scores[i] if 0 <= i < len(scores) else 0.0,
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for idx in ranked[:top_n]:
        if idx < 0 or idx >= len(scores):
            continue
        score = float(scores[idx])
        if score <= 0:
            continue
        out.append({"index": int(idx), "score": score, "meta": metadata[idx]})
    return out


def _match_any_pattern(value: str, patterns: List[str]) -> bool:
    if not value:
        return False
    value_key = _normalize_for_match(value)
    for pattern in patterns:
        p_key = _normalize_for_match(pattern)
        if not p_key:
            continue
        if p_key in value_key or value_key in p_key:
            return True
    return False


def _detect_postgrad_target_program(query: str) -> Optional[str]:
    q_key = _normalize_for_match(query)
    for program_key, aliases in POSTGRAD_PROGRAM_QUERY_ALIASES.items():
        for alias in aliases:
            if _normalize_for_match(alias) in q_key:
                return program_key
    return None


def _is_postgrad_query(query: str) -> bool:
    q_key = _normalize_for_match(query)
    return any(hint in q_key for hint in POSTGRAD_QUERY_HINTS)


def apply_query_source_priority(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not results:
        return results

    target_program = _detect_postgrad_target_program(query)
    if not _is_postgrad_query(query) and target_program is None:
        return results

    boosted: List[Dict[str, Any]] = []
    for item in results:
        source = _source_from_item(item)
        multiplier = 1.0

        if source:
            if target_program and _match_any_pattern(source, POSTGRAD_PROGRAM_SOURCES[target_program]):
                multiplier = 14.0
            elif _match_any_pattern(source, ALL_POSTGRAD_GUIDE_SOURCES):
                multiplier = 7.0
            elif _match_any_pattern(source, POSTGRAD_GENERAL_SOURCES):
                multiplier = 3.0

        score = float(item.get("score", 0.0))
        if multiplier > 1.0:
            item["score"] = score * multiplier + 0.01 * (multiplier - 1.0)

        boosted.append(item)

    boosted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return boosted


def build_query_for_embedding(q: str) -> str:
    # E5-style prefix for queries
    q = normalize_query(q)
    return f"query: {q}"


def _trim_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ----------------------
# Search / Retrieval
# ----------------------
def search(query: str,
           model: Any,
           faiss_index: Optional[Any],
           embeddings: Optional[np.ndarray],
           metadata: List[Dict[str, Any]],
           top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return top-k results as list of dicts: {score, metadata}.

    Uses FAISS if available, otherwise numpy dot-product on embeddings.
    Assumes embeddings and query embeddings are L2-normalized when using cosine/IP.
    """
    q_emb = model.encode([build_query_for_embedding(query)], normalize_embeddings=True, convert_to_numpy=True)[0]

    results = []
    if faiss_index is not None:
        # FAISS expects shape (n, d)
        q = q_emb.astype(np.float32).reshape(1, -1)
        D, I = faiss_index.search(q, top_k)
        # D: distances/scores, I: indices
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            results.append({"score": float(score), "meta": metadata[idx]})
        return results

    if embeddings is None:
        raise RuntimeError("No FAISS index and no embeddings array available")

    # numpy dot product fallback (embeddings expected normalized)
    sims = embeddings @ q_emb
    ids = np.argsort(-sims)[:top_k]
    for i in ids:
        results.append({"score": float(sims[i]), "meta": metadata[int(i)]})
    return results


def format_results(results: List[Dict[str, Any]]) -> str:
    out_lines = []
    for r in results:
        score = r["score"]
        meta = r.get("meta") if isinstance(r, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        text = _trim_context(meta.get("retrieval_text") or meta.get("text") or "", MAX_CONTEXT_CHARS)
        out_lines.append(f"score={score:.4f} source={meta.get('source')} section={meta.get('section')}\n{text}\n----")
    return "\n".join(out_lines)


# ----------------------
# Extended pipeline functions
# ----------------------
def embed_query(model: Any, query: str) -> np.ndarray:
    """Return normalized embedding vector for the query."""
    q_text = build_query_for_embedding(query)
    emb = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return emb


def search_index(q_emb: np.ndarray,
                 faiss_index: Optional[Any],
                 embeddings: Optional[np.ndarray],
                 top_k: int = TOP_K,
                 allowed_indices: Optional[List[int]] = None) -> Tuple[List[int], List[float]]:
    """Return (indices, scores) from FAISS or numpy fallback."""
    if allowed_indices is not None:
        if not allowed_indices:
            return [], []
        if embeddings is None:
            raise RuntimeError("Embeddings are required for strict filtered dense retrieval")

        idx_array = np.array(allowed_indices, dtype=np.int64)
        sub_embeddings = embeddings[idx_array]
        sims = sub_embeddings @ q_emb
        local_top = np.argsort(-sims)[:top_k]
        global_indices = [int(idx_array[i]) for i in local_top.tolist()]
        global_scores = [float(sims[i]) for i in local_top.tolist()]
        return global_indices, global_scores

    if faiss_index is not None:
        q = q_emb.astype(np.float32).reshape(1, -1)
        D, I = faiss_index.search(q, top_k)
        return I[0].tolist(), D[0].tolist()

    if embeddings is None:
        raise RuntimeError("No index available")
    sims = embeddings @ q_emb
    idxs = np.argsort(-sims)[:top_k]
    return idxs.tolist(), sims[idxs].tolist()


def build_results(metadata: List[Dict[str, Any]], indices: List[int], scores: List[float]) -> List[Dict[str, Any]]:
    out = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(metadata):
            continue
        out.append({"index": int(idx), "score": float(score), "meta": metadata[int(idx)]})
    return out


def _is_similar(a: str, b: str, thresh: float = 0.85) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= thresh


def filter_results(results: List[Dict[str, Any]],
                   min_score: float = SCORE_THRESHOLD,
                   duplicate_thresh: float = 0.85,
                   min_chars: int = 30) -> List[Dict[str, Any]]:
    """Filter by score, remove duplicates and tiny/empty chunks."""
    filtered = []
    seen_texts: List[str] = []
    for r in results:
        score = float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0
        if score < min_score:
            continue

        meta = r.get("meta") if isinstance(r, dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        # Some fused candidates may not carry metadata; ignore non-informative chunks.
        text = (
            meta.get("retrieval_text")
            or meta.get("text")
            or (r.get("retrieval_text") if isinstance(r, dict) else "")
            or (r.get("text") if isinstance(r, dict) else "")
            or ""
        ).strip()
        if len(text) < min_chars:
            continue
        dup = False
        for s in seen_texts:
            if _is_similar(s, text, duplicate_thresh):
                dup = True
                break
        if dup:
            continue
        seen_texts.append(text)
        filtered.append(r)
    return filtered


def rerank_with_cross_encoder(cross_encoder_model_name: str, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        logger.info("CrossEncoder not available, skipping rerank")
        return candidates
    model = CrossEncoder(cross_encoder_model_name)
    pairs = [[query, (c["meta"].get("retrieval_text") or c["meta"].get("text"))] for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda x: -x["score"])[:TOP_K]


def merge_chunks(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, int, bool]:
    parts = []
    total = 0
    included = 0
    truncated = False
    for c in chunks:
        meta = c.get("meta") if isinstance(c, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        text = (
            meta.get("retrieval_text")
            or meta.get("text")
            or (c.get("retrieval_text") if isinstance(c, dict) else "")
            or (c.get("text") if isinstance(c, dict) else "")
            or ""
        )
        if not text:
            continue
        remain = max_chars - total
        if remain <= 0:
            truncated = True
            break
        piece = text if len(text) <= remain else text[:remain] + "..."
        if len(text) > remain:
            truncated = True
        parts.append(f"[Source] {meta.get('source')}\n{piece}")
        total += len(piece)
        included += 1
    return "\n\n".join(parts), included, truncated


def build_prompt_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, int, bool]:
    return merge_chunks(chunks, max_chars)


def load_or_build_bm25(out_dir: Path,
                       metadata: List[Dict[str, Any]],
                       filename: str = BM25_INDEX_FILENAME) -> BM25Index:
    idx_path = out_dir / filename
    logger.info("Loading or building BM25 index at %s", idx_path)
    return load_or_build_bm25_index(idx_path, metadata)


def retrieve(query: str,
             model: Any,
             faiss_index: Optional[Any],
             embeddings: Optional[np.ndarray],
             metadata: List[Dict[str, Any]],
             bm25_index: Optional[BM25Index] = None,
             top_k: int = TOP_K,
             dense_top_n: int = HYBRID_DENSE_TOP_N,
             sparse_top_n: int = HYBRID_SPARSE_TOP_N,
             rrf_k: int = RRF_K,
             min_score: float = SCORE_THRESHOLD,
             duplicate_thresh: float = 0.85,
             enable_program_filter_fallback: bool = False,
             min_filtered_chunks_for_fallback: int = 5,
             min_chunks_for_context: int = 15,
             max_chunks_for_context: int = 20,
             context_max_chars: int = MAX_CONTEXT_CHARS) -> Dict[str, Any]:
    """Run full retrieval pipeline and return final context + raw results."""
    q_norm = normalize_query(query)
    analysis = analyze_query(q_norm)
    detected_program = analysis.get("program")
    category = str(analysis.get("category") or "general")

    allowed_indices: Optional[List[int]] = None
    fallback_used = False
    filtered_corpus_size = len(metadata)

    if detected_program is not None:
        allowed_indices = filter_metadata_by_program(metadata, detected_program)
        filtered_corpus_size = len(allowed_indices)

        if (
            enable_program_filter_fallback
            and filtered_corpus_size < min_filtered_chunks_for_fallback
        ):
            allowed_indices = None
            filtered_corpus_size = len(metadata)
            fallback_used = True

    q_emb = embed_query(model, q_norm)

    dense_pool = max(top_k * 4, dense_top_n)
    if allowed_indices is not None:
        dense_pool = min(dense_pool, max(len(allowed_indices), 1))

    indices, scores = search_index(
        q_emb,
        faiss_index,
        embeddings,
        top_k=dense_pool,
        allowed_indices=allowed_indices,
    )
    dense_raw = build_results(metadata, indices, scores)

    retrieval_mode = "dense"
    raw = dense_raw
    effective_min_score = min_score

    if bm25_index is not None:
        sparse_raw = filtered_bm25_search(
            bm25_index,
            q_norm,
            metadata,
            top_n=sparse_top_n,
            allowed_indices=allowed_indices,
        )
        fused = reciprocal_rank_fusion(
            [dense_raw[:dense_top_n], sparse_raw],
            rrf_k=rrf_k,
        )
        raw = fused
        retrieval_mode = "hybrid_rrf"
        # RRF scores are rank-based and on a small range, so thresholding is disabled.
        effective_min_score = 0.0

    raw = apply_query_source_priority(raw, q_norm)
    raw = apply_bonus_rerank(raw, category=category, detected_program=detected_program)

    filtered = filter_results(raw, min_score=effective_min_score, duplicate_thresh=duplicate_thresh)
    # Select at least min_chunks_for_context (if available) and cap at max_chunks_for_context.
    effective_max_chunks = max(max_chunks_for_context, min_chunks_for_context)
    final_k = min(effective_max_chunks, max(min_chunks_for_context, top_k))
    final = filtered[:final_k]
    context, context_chunk_count, context_truncated = build_prompt_context(final, max_chars=context_max_chars)
    return {
        "query": query,
        "results": final,
        "context": context,
        "mode": retrieval_mode,
        "query_analysis": {
            "program": detected_program,
            "category": category,
            "fallback_used": fallback_used,
        },
        "filtered_corpus_size": filtered_corpus_size,
        "requested_top_k": top_k,
        "final_result_count": len(final),
        "context_chunk_count": context_chunk_count,
        "context_truncated": context_truncated,
        "context_char_len": len(context),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick retrieval CLI")
    parser.add_argument("--out", type=str, default="../Embedding/out", help="output folder with embeddings/metadata/faiss")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    out_dir = Path(args.out)
    meta = load_metadata(out_dir / METADATA_FILENAME)
    faiss_idx = None
    embeddings = None
    if _HAS_FAISS and (out_dir / FAISS_INDEX_FILENAME).exists():
        faiss_idx = load_faiss_index(out_dir / FAISS_INDEX_FILENAME)
    else:
        # try load numpy embeddings
        embeddings = load_embeddings_np(out_dir / EMBEDDINGS_FILENAME)

    model = load_embedding_model()
    results = search(args.query, model, faiss_idx, embeddings, meta, top_k=args.k)
    print(format_results(results))
