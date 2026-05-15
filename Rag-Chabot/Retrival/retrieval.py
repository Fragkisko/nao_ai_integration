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
from typing import List, Dict, Any, Optional

import numpy as np
from difflib import SequenceMatcher

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
MAX_CONTEXT_CHARS = 1000

EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
FAISS_INDEX_FILENAME = "faiss.index"

# Rerank / Cross-encoder defaults
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20  # top-N candidates to rerank
RERANK_ENABLED = True

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
           model: SentenceTransformer,
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
        meta = r["meta"]
        text = _trim_context(meta.get("retrieval_text") or meta.get("text") or "", MAX_CONTEXT_CHARS)
        out_lines.append(f"score={score:.4f} source={meta.get('source')} section={meta.get('section')}\n{text}\n----")
    return "\n".join(out_lines)


# ----------------------
# Extended pipeline functions
# ----------------------
def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Return normalized embedding vector for the query."""
    q_text = build_query_for_embedding(query)
    emb = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return emb


def search_index(q_emb: np.ndarray,
                 faiss_index: Optional[Any],
                 embeddings: Optional[np.ndarray],
                 top_k: int = TOP_K):
    """Return (indices, scores) from FAISS or numpy fallback."""
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
        if r["score"] < min_score:
            continue
        text = (r["meta"].get("retrieval_text") or r["meta"].get("text") or "").strip()
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
    # Use prefixes to match training format (query:/passage:) and normalize inputs
    q_norm = normalize_query(query)
    pairs = [
        [f"query: {q_norm}", f"passage: {normalize_query((c['meta'].get('retrieval_text') or c['meta'].get('text') or ''))}"]
        for c in candidates
    ]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda x: -x["score"])[:TOP_K]


def merge_chunks(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for c in chunks:
        text = c["meta"].get("retrieval_text") or c["meta"].get("text") or ""
        if not text:
            continue
        remain = max_chars - total
        if remain <= 0:
            break
        piece = text if len(text) <= remain else text[:remain] + "..."
        parts.append(f"[Source] {c['meta'].get('source')}\n{piece}")
        total += len(piece)
    return "\n\n".join(parts)


def build_prompt_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    return merge_chunks(chunks, max_chars)


def retrieve(query: str,
             model: SentenceTransformer,
             faiss_index: Optional[Any],
             embeddings: Optional[np.ndarray],
             metadata: List[Dict[str, Any]],
             top_k: int = TOP_K,
             min_score: float = SCORE_THRESHOLD,
             duplicate_thresh: float = 0.85) -> Dict[str, Any]:
    """Run full retrieval pipeline and return final context + raw results."""
    q_norm = normalize_query(query)
    q_emb = embed_query(model, q_norm)
    indices, scores = search_index(q_emb, faiss_index, embeddings, top_k=top_k*4)
    raw = build_results(metadata, indices, scores)
    filtered = filter_results(raw, min_score=min_score, duplicate_thresh=duplicate_thresh)

    # Optional rerank: rerank top-N candidates with a cross-encoder, then take top_k
    if RERANK_ENABLED and filtered:
        try:
            logger.info("Reranking top %d candidates with %s", RERANK_CANDIDATES, CROSS_ENCODER_MODEL)
            candidates = filtered[:RERANK_CANDIDATES]
            reranked = rerank_with_cross_encoder(CROSS_ENCODER_MODEL, query, candidates)
            final = reranked[:top_k]
        except Exception as e:
            logger.exception("Reranking failed, falling back to original ranking: %s", e)
            final = filtered[:top_k]
    else:
        final = filtered[:top_k]

    context = build_prompt_context(final)
    return {"query": query, "results": final, "context": context}


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
