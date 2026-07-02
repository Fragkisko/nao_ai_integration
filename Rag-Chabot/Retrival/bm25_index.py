"""BM25 sparse index utilities for chunk metadata.

This module is intentionally independent from dense retrieval (FAISS/embeddings).
It can build a BM25 index from metadata chunks, persist it to disk, and query it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import re
import unicodedata


def _normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    # Keep alphanumeric words (works for Greek/Latin with UNICODE flag).
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _chunk_text(meta: Dict[str, Any]) -> str:
    return (meta.get("retrieval_text") or meta.get("text") or "").strip()


@dataclass
class BM25Index:
    k1: float
    b: float
    doc_count: int
    avgdl: float
    doc_lens: List[int]
    doc_tfs: List[Dict[str, int]]
    dfs: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "doc_lens": self.doc_lens,
            "doc_tfs": self.doc_tfs,
            "dfs": self.dfs,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BM25Index":
        return cls(
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
            doc_count=int(payload.get("doc_count", 0)),
            avgdl=float(payload.get("avgdl", 0.0)),
            doc_lens=[int(x) for x in payload.get("doc_lens", [])],
            doc_tfs=[{str(k): int(v) for k, v in d.items()} for d in payload.get("doc_tfs", [])],
            dfs={str(k): int(v) for k, v in payload.get("dfs", {}).items()},
        )

    def _idf(self, term: str) -> float:
        # BM25 idf with +1 for numerical stability.
        df = self.dfs.get(term, 0)
        if df <= 0:
            return 0.0
        num = (self.doc_count - df + 0.5)
        den = (df + 0.5)
        return math.log(1.0 + (num / den))

    def score_query(self, query: str) -> List[float]:
        tokens = _tokenize(query)
        if not tokens or self.doc_count == 0:
            return [0.0] * self.doc_count

        scores = [0.0] * self.doc_count
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for term in tokens:
            idf = self._idf(term)
            if idf <= 0:
                continue

            for doc_id in range(self.doc_count):
                tf = self.doc_tfs[doc_id].get(term, 0)
                if tf <= 0:
                    continue

                dl = self.doc_lens[doc_id] if doc_id < len(self.doc_lens) else 0
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                if denom <= 0:
                    continue
                scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / denom)

        return scores

    def search(self,
               query: str,
               top_n: int,
               metadata: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        scores = self.score_query(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Dict[str, Any]] = []
        for idx in ranked[:top_n]:
            score = float(scores[idx])
            if score <= 0:
                continue
            item = {
                "index": int(idx),
                "score": score,
            }
            if metadata is not None and 0 <= idx < len(metadata):
                item["meta"] = metadata[idx]
            out.append(item)
        return out


def build_bm25_index(metadata: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    doc_tfs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    dfs: Dict[str, int] = {}

    for meta in metadata:
        text = _chunk_text(meta)
        tokens = _tokenize(text)
        doc_lens.append(len(tokens))

        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        doc_tfs.append(tf)

        for tok in tf.keys():
            dfs[tok] = dfs.get(tok, 0) + 1

    doc_count = len(metadata)
    avgdl = (sum(doc_lens) / doc_count) if doc_count > 0 else 0.0

    return BM25Index(
        k1=k1,
        b=b,
        doc_count=doc_count,
        avgdl=avgdl,
        doc_lens=doc_lens,
        doc_tfs=doc_tfs,
        dfs=dfs,
    )


def save_bm25_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, ensure_ascii=False)


def load_bm25_index(path: Path) -> BM25Index:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return BM25Index.from_dict(payload)


def load_or_build_bm25_index(path: Path,
                             metadata: List[Dict[str, Any]],
                             k1: float = 1.5,
                             b: float = 0.75) -> BM25Index:
    if path.exists():
        return load_bm25_index(path)

    index = build_bm25_index(metadata, k1=k1, b=b)
    save_bm25_index(index, path)
    return index
