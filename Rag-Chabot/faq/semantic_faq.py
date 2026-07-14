from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
import contextlib
import io
from typing import Any, Dict, List, Optional

import numpy as np


def _load_retrieval_module(rag_root: Path):
    retrieval_path = rag_root / "Retrival" / "retrieval.py"
    retrieval_dir = str(retrieval_path.parent.resolve())
    if retrieval_dir not in sys.path:
        sys.path.insert(0, retrieval_dir)
    spec = importlib.util.spec_from_file_location("retrieval_for_faq", str(retrieval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load retrieval module from {retrieval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SemanticFAQSearcher:
    def __init__(
        self,
        faq_dir: Path,
        threshold: float = 0.88,
        model_name: Optional[str] = None,
        preload_model: bool = False,
    ) -> None:
        self.faq_dir = Path(faq_dir)
        self.threshold = float(threshold)
        self.rag_root = Path(__file__).resolve().parents[1]
        self.retrieval = _load_retrieval_module(self.rag_root)

        self.model_name = model_name or getattr(
            self.retrieval,
            "EMBEDDING_MODEL_NAME",
            "intfloat/multilingual-e5-base",
        )

        self.metadata_path = self.faq_dir / "metadata.json"
        self.embeddings_path = self.faq_dir / "embeddings.npy"
        self.faiss_path = self.faq_dir / "faiss.index"

        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None
        self.model: Optional[Any] = None

        self._load_index()
        if preload_model:
            self._ensure_model_loaded()

    def _load_index(self) -> None:
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as fp:
                self.metadata = json.load(fp)

        if self.embeddings_path.exists():
            self.embeddings = self.retrieval.load_embeddings_np(self.embeddings_path)

        if getattr(self.retrieval, "_HAS_FAISS", False) and self.faiss_path.exists():
            self.faiss_index = self.retrieval.load_faiss_index(self.faiss_path)

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            # Keep model loading quiet to avoid progress output during live dialog.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.model = self.retrieval.load_embedding_model(self.model_name)

    def is_ready(self) -> bool:
        if not self.metadata:
            return False
        return self.faiss_index is not None or self.embeddings is not None

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        q = (query or "").strip()
        if not q or not self.is_ready():
            return None

        self._ensure_model_loaded()
        q_norm = self.retrieval.normalize_query(q)
        q_emb = self.retrieval.embed_query(self.model, q_norm)

        indices, scores = self.retrieval.search_index(
            q_emb,
            self.faiss_index,
            self.embeddings,
            top_k=1,
        )
        if not indices:
            return None

        best_idx = int(indices[0])
        best_score = float(scores[0]) if scores else 0.0
        if best_idx < 0 or best_idx >= len(self.metadata):
            return None

        faq_item = self.metadata[best_idx]
        if best_score < self.threshold:
            return None

        return {
            "answer": (faq_item.get("answer") or "").strip(),
            "score": best_score,
            "matched_faq": faq_item,
            "threshold": self.threshold,
        }
