#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys
from typing import Any, Dict, List


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dependencies(rag_root: Path):
    retrieval_path = rag_root / "Retrival" / "retrieval.py"
    embedding_path = rag_root / "Embedding" / "embeddings.py"

    retrieval_dir = str(retrieval_path.parent.resolve())
    if retrieval_dir not in sys.path:
        sys.path.insert(0, retrieval_dir)

    retrieval = _load_module(retrieval_path, "retrieval_for_faq_build")
    embeddings = _load_module(embedding_path, "embeddings_for_faq_build")
    return retrieval, embeddings


def _load_faq_items(faq_json_path: Path) -> List[Dict[str, Any]]:
    with faq_json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, list):
        raise RuntimeError("faq.json must be a list of FAQ objects")

    normalized: List[Dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue

        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            continue

        normalized.append(
            {
                "id": raw.get("id"),
                "question": question,
                "answer": answer,
                "category": raw.get("category"),
            }
        )

    if not normalized:
        raise RuntimeError("No valid FAQ entries found in faq.json")

    return normalized


def build_faq_index(
    faq_json_path: Path,
    out_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    rag_root = Path(__file__).resolve().parents[1]
    retrieval, embeddings = _load_dependencies(rag_root)

    items = _load_faq_items(faq_json_path)

    texts = [f"passage: {item['question']}" for item in items]

    embs = embeddings.embed_texts(model_name, texts, batch_size=batch_size).astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings.save_embeddings_numpy(out_dir, embs, items)

    if getattr(embeddings, "_HAS_FAISS", False):
        embeddings.build_faiss_index(out_dir, embs.copy(), items, metric="cosine")

    print(f"FAQ index built successfully at: {out_dir}")
    print(f"Entries indexed: {len(items)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic FAQ index")
    parser.add_argument(
        "--faq-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "faq.json"),
        help="Path to faq.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Output folder for FAQ index artifacts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Embedding model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_faq_index(
        faq_json_path=Path(args.faq_json),
        out_dir=Path(args.out),
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
