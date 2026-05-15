try:
    import faiss
except Exception:
    faiss = None

from typing import List, Union

class VectorStore:
    """Simple FAISS-backed vector store wrapper (skeleton)."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        if faiss is None:
            self.index = None
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        self.docs: List[str] = []

    def index_documents(self, embeddings, docs: List[str]):
        if self.index is None:
            raise RuntimeError("faiss is not available")
        import numpy as np
        self.docs = docs
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query: Union[str, List[float], "np.ndarray"], top_k: int = 5) -> List[str]:
        # This skeleton accepts either a pre-computed embedding or raises if a raw string is provided.
        if self.index is None:
            return []
        import numpy as np
        if isinstance(query, str):
            raise RuntimeError("search expects an embedding; encode the text first")
        query_np = np.array([query]).astype("float32")
        distances, idxs = self.index.search(query_np, top_k)
        return [self.docs[i] for i in idxs[0] if i != -1 and i < len(self.docs)]
