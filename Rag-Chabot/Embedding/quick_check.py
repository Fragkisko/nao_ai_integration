import json, numpy as np, argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="out")
parser.add_argument("--query", default=None)
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

out = Path(args.out)
emb = np.load(out / "embeddings.npy")
meta = json.load(open(out / "metadata.json", encoding="utf-8"))

print("emb shape:", emb.shape)
if args.query:
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    q_text = f"query: {args.query}"
    q_emb = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = emb @ q_emb   # dot product -> cosine if embeddings normalized
    idxs = sims.argsort()[::-1][: args.k]
    for i in idxs:
        print(f"score={sims[i]:.4f} source={meta[i].get('source')} section={meta[i].get('section')}")
        print(meta[i].get("retrieval_text")[:400].replace("\n"," ") + "...")
        print("----")
else:
    print("No query provided. Use --query \"your question\"")