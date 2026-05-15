import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers")

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def load_preview(preview_path: Path) -> List[Dict[str, Any]]:
    with open(preview_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def collect_chunks_from_dir(chunks_dir: Path) -> List[Dict[str, Any]]:
    # Fallback: read all *_chunk_*.txt files
    items = []
    files = sorted(chunks_dir.glob("*_chunk_*.txt"))
    idx = 1
    for p in files:
        text = p.read_text(encoding="utf-8")
        items.append({
            "id": idx,
            "text": text,
            "source": p.name,
            "chunk_index": idx,
            "section": None,
            "start_char": 0,
            "end_char": len(text),
            "language": "el",
        })
        idx += 1
    return items


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("source"), it.get("chunk_index"), it.get("text"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def embed_texts(model_name: str, texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs


def save_embeddings_numpy(output_dir: Path, embs: np.ndarray, metadata: List[Dict[str, Any]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embs)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_faiss_index(output_dir: Path, embs: np.ndarray, metadata: List[Dict[str, Any]], metric: str = "cosine"):
    if not _HAS_FAISS:
        raise RuntimeError("faiss is not installed")
    d = embs.shape[1]
    if metric == "cosine":
        # normalize vectors and use inner product
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(embs)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build embeddings for chunks using multilingual-e5-base")
    parser.add_argument("--preview", type=str, help="Path to chunk_preview.json (preferred)")
    parser.add_argument("--chunks-dir", type=str, help="Fallback directory with chunk txt files")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base", help="HF model name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="./embeddings_out")
    parser.add_argument("--use-faiss", action="store_true", help="Also build FAISS index if faiss available")
    args = parser.parse_args()

    out_dir = Path(args.output)

    items = []
    if args.preview:
        preview_path = Path(args.preview)
        if not preview_path.exists():
            raise SystemExit(f"Preview file not found: {preview_path}")
        items = load_preview(preview_path)
    elif args.chunks_dir:
        items = collect_chunks_from_dir(Path(args.chunks_dir))
    else:
        raise SystemExit("Provide --preview or --chunks-dir")

    items = dedupe_items(items)
    # enrich items with retrieval_text (E5-style) and build texts with E5 prefix
    texts = []
    for it in items:
        retrieval_text = (
            f"Section: {it.get('section')}\n\n"
            f"Source: {it.get('source')}\n\n"
            "Content:\n"
            f"{it.get('text')}"
        )
        # keep retrieval_text in metadata for later use
        it["retrieval_text"] = retrieval_text
        # E5 expects passage prefix for passages
        texts.append(f"passage: {retrieval_text}")

    if not texts:
        raise SystemExit("No chunks found to embed")

    embs = embed_texts(args.model, texts, batch_size=args.batch_size)

    # If cosine metric, store normalized embeddings for FAISS cosine usage
    if args.use_faiss and _HAS_FAISS:
        # FAISS expects float32
        embs = embs.astype(np.float32)
        build_faiss_index(out_dir, embs.copy(), items, metric="cosine")
    else:
        # Save raw embeddings + metadata
        save_embeddings_numpy(out_dir, embs.astype(np.float32), items)

    print(f"Saved embeddings and metadata to {out_dir}")


if __name__ == "__main__":
    main()
