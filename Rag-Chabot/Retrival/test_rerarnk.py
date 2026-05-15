# test_rerank.py
from sentence_transformers import SentenceTransformer
from retrieval import retrieve, load_metadata  # adjust import path if χρειάζεται
import numpy as np

# small synthetic corpus
metadata = [
    {"source":"doc1","section":"s1","retrieval_text":"Ο καιρός σήμερα είναι ηλιόλουστος."},
    {"source":"doc2","section":"s2","retrieval_text":"Η οικονομία παρουσιάζει ανάπτυξη."},
    {"source":"doc3","section":"s3","retrieval_text":"Συνταγές για παραδοσιακή χωριάτικη σαλάτα."},
]

# load embedding model (E5)
model = SentenceTransformer("intfloat/multilingual-e5-base")

# create embeddings for passages (normalized)
texts = ["passage: " + (m["retrieval_text"]) for m in metadata]
embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

# run retrieve (uses RERANK_ENABLED, CROSS_ENCODER_MODEL defaults from retrieval.py)
res = retrieve(
    query="Τι καιρό θα κάνει σήμερα;",
    model=model,
    faiss_index=None,
    embeddings=embs,
    metadata=metadata,
    top_k=3
)

print("Context:\n", res["context"])
print("Results:")
for r in res["results"]:
    print(f"- score={r['score']:.4f} source={r['meta']['source']} text={r['meta']['retrieval_text']}")