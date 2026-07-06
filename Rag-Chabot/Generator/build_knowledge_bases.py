#!/usr/bin/env python3
"""Build independent bachelor/master knowledge bases using the existing chunking and embedding logic."""
from pathlib import Path
import argparse
import json
import sys


RAG_ROOT = Path(__file__).resolve().parents[1]
CHUNKING_DIR = RAG_ROOT / "Chunking"
EMBEDDING_DIR = RAG_ROOT / "Embedding"

for module_dir in (CHUNKING_DIR, EMBEDDING_DIR):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

import chunker  # type: ignore
import embeddings  # type: ignore


MODE_CONFIG = {
    "bachelor": {
        "source_dir": RAG_ROOT / "Documents" / "Bachelor",
        "output_dir": RAG_ROOT / "knowledge_base" / "bachelor",
    },
    "master": {
        "source_dir": RAG_ROOT / "Documents" / "Master",
        "output_dir": RAG_ROOT / "knowledge_base" / "master",
    },
}


def build_chunks(raw_text_dir: Path, chunk_dir: Path, *, max_chars: int, max_words, spacy_model: str, verbose: bool) -> Path:
    nlp = chunker.spacy.load(spacy_model)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_text_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {raw_text_dir}")

    chunk_preview = []
    global_chunk_id = 1

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunker.chunk_text_with_spacy(
            text,
            nlp,
            max_chars=max_chars,
            max_words=max_words,
            overlap_sents=1,
            verbose=verbose,
        )
        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data["text"]
            out_file = chunk_dir / f"{txt_file.stem}_chunk_{i + 1}.txt"
            out_file.write_text(chunk_text, encoding="utf-8")
            chunk_preview.append(
                {
                    "id": global_chunk_id,
                    "text": chunk_text,
                    "retrieval_text": chunk_data["retrieval_text"],
                    "source": txt_file.name,
                    "chunk_index": i + 1,
                    "start_char": chunk_data["start"],
                    "end_char": chunk_data["end"],
                    "section": chunk_data["section"],
                    "language": "el",
                }
            )
            global_chunk_id += 1
        print(f"Chunked {txt_file.name} into {len(chunks)} chunks. Saved to {chunk_dir}")

    preview_path = chunk_dir / "chunk_preview.json"
    with preview_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_preview, f, ensure_ascii=False, indent=2)
    print(f"Chunk preview file written to {preview_path}")
    return preview_path


def build_embeddings(preview_path: Path, output_dir: Path, *, model_name: str, batch_size: int, use_faiss: bool) -> None:
    items = embeddings.dedupe_items(embeddings.load_preview(preview_path))

    texts = []
    for item in items:
        retrieval_text = (
            f"Section: {item.get('section')}\n\n"
            f"Source: {item.get('source')}\n\n"
            "Content:\n"
            f"{item.get('text')}"
        )
        item["retrieval_text"] = retrieval_text
        texts.append(f"passage: {retrieval_text}")

    if not texts:
        raise RuntimeError(f"No chunks found to embed from {preview_path}")

    embs = embeddings.embed_texts(model_name, texts, batch_size=batch_size).astype("float32")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep embeddings.npy available even when FAISS is built, because strict
    # filtered dense retrieval still uses the embeddings array for restricted searches.
    embeddings.save_embeddings_numpy(output_dir, embs, items)

    if use_faiss and getattr(embeddings, "_HAS_FAISS", False):
        embeddings.build_faiss_index(output_dir, embs.copy(), items, metric="cosine")

    print(f"Saved knowledge base to {output_dir}")


def build_mode(mode: str, *, force: bool, max_chars: int, max_words, spacy_model: str, embedding_model: str, batch_size: int, use_faiss: bool, verbose: bool) -> None:
    config = MODE_CONFIG[mode]
    source_dir = config["source_dir"]
    output_dir = config["output_dir"]
    raw_text_dir = output_dir / "raw_texts"
    chunk_dir = output_dir / "chunks"

    print(f"Building knowledge base: {mode}")
    chunker.process_directory(source_dir, raw_text_dir, overwrite=force)
    preview_path = build_chunks(
        raw_text_dir,
        chunk_dir,
        max_chars=max_chars,
        max_words=max_words,
        spacy_model=spacy_model,
        verbose=verbose,
    )
    build_embeddings(
        preview_path,
        output_dir,
        model_name=embedding_model,
        batch_size=batch_size,
        use_faiss=use_faiss,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bachelor/master knowledge bases with the existing RAG pipeline")
    parser.add_argument("--mode", choices=["bachelor", "master", "all"], default="all")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite extracted raw text files")
    parser.add_argument("--max-chars", type=int, default=300, help="Max characters per chunk")
    parser.add_argument("--max-words", type=int, default=None, help="Optional max words per chunk")
    parser.add_argument("--spacy-model", type=str, default="el_core_news_sm", help="spaCy model for sentence splitting")
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-base", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-faiss", action="store_true", help="Build FAISS index when faiss is available")
    parser.add_argument("--verbose", action="store_true", help="Print chunking details")
    args = parser.parse_args()

    modes = [args.mode] if args.mode != "all" else ["bachelor", "master"]
    for mode in modes:
        build_mode(
            mode,
            force=args.force,
            max_chars=args.max_chars,
            max_words=args.max_words,
            spacy_model=args.spacy_model,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            use_faiss=args.use_faiss,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()