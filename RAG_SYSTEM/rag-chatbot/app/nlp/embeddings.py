try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class SentenceEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
