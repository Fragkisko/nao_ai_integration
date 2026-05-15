#!/usr/bin/env python3
"""Simple RAG CLI: retrieval -> prompt -> OpenAI LLM.

Usage:
    python rag_cli.py --out ../Embedding/out --model gpt-4o-mini

Interactive mode if --query not provided.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import requests
import sys
from time import time
import logging
import os
import textwrap


def _load_dotenv_from_parents(start_dir: Path) -> None:
    """Best-effort .env loader without external dependency.

    Looks for a .env file in current/parent directories and injects values
    into process environment only when they are not already defined.
    """
    current = start_dir.resolve()
    while True:
        env_path = current / ".env"
        if env_path.exists():
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key:
                            os.environ.setdefault(key, value)
            except Exception:
                pass
            return

        if current.parent == current:
            return
        current = current.parent


def load_retrieval_module(rag_root: Path):
    # retrieval is in Rag-Chabot/Retrival/retrieval.py
    retrieval_path = rag_root / "Retrival" / "retrieval.py"
    spec = importlib.util.spec_from_file_location("retrieval", str(retrieval_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_structured_retrieval_module(rag_root: Path):
    structured_path = rag_root / "Retrival" / "structured_retrieval.py"
    spec = importlib.util.spec_from_file_location("structured_retrieval", str(structured_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def openai_chat(api_base_url: str, api_key: str, model: str, prompt: str, temperature: float = 0.0):
    url = api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer precisely."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    # OpenAI chat.completions shape: {'choices': [{'message': {'content': '...'}}]}
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Defensive handling for multimodal-style content blocks.
                    out = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            out.append(item.get("text", ""))
                    return "".join(out).strip()
                return str(content).strip()
        return json.dumps(data)

    return str(data)


def build_prompt_from_context(context: str, query: str) -> str:
    prompt = (
        "Παρακάτω είναι τα σχετικά αποσπάσματα από τα έγγραφα. Χρησιμοποίησέ τα για να απαντήσεις στην ερώτηση.\n\n"
        f"Context:\n{context}\n\n"
        f"Ερώτηση: {query}\n\n"
        "Απάντησε σύντομα με υφος ξεναγου και μην προσθέτεις τίποτα που δεν μπορείς να τεκμηριώσεις από τα αποσπάσματα. Αν δεν ξέρεις, πες το ευθέως."
    )
    return prompt


def build_prompt_from_structured_facts(query: str, structured_answer: dict) -> str:
    source_table = structured_answer.get("source_table")
    facts = structured_answer.get("row") or {}
    deterministic = structured_answer.get("answer") or ""
    facts_json = json.dumps(facts, ensure_ascii=False)

    return (
        "Θα απαντήσεις σε φοιτητή με φυσικό, σύντομο, ανθρώπινο ύφος.\n"
        "Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ τα FACTS που δίνονται παρακάτω.\n"
        "Απαγορεύεται να προσθέσεις πληροφορία που δεν υπάρχει στα FACTS.\n"
        "Αν λείπει πεδίο, πες ότι δεν υπάρχει διαθέσιμη πληροφορία.\n"
        "Μέγιστο 3 προτάσεις.\n\n"
        f"Πηγή: {source_table}\n"
        f"FACTS(JSON): {facts_json}\n"
        f"Προκατασκευασμένη ασφαλής απάντηση: {deterministic}\n\n"
        f"Ερώτηση χρήστη: {query}\n"
        "Δώσε τελική απάντηση στα ελληνικά."
    )


def render_structured_answer(query: str, structured_answer: dict, args) -> str:
    if args.structured_mode == "raw":
        return (structured_answer.get("answer") or "").strip()

    prompt = build_prompt_from_structured_facts(query, structured_answer)
    try:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")
        llm_answer = openai_chat(args.api_base_url, api_key, args.model, prompt, temperature=0.2).strip()
        if llm_answer:
            return llm_answer
    except Exception:
        pass

    # deterministic fallback when LLM is down or returns empty output
    return (structured_answer.get("answer") or "").strip()


def main():
    # Ensure env vars from repo-level .env are available when script is launched
    # from nested folders (e.g. Rag-Chabot/Generator).
    _load_dotenv_from_parents(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="../Embedding/out", help="embedding output folder")
    parser.add_argument("--db-path", type=str, default="../IEE_IHU_DB/NAO_IEE_IHU_DB.db", help="SQLite DB path for structured staff answers")
    parser.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable that contains the OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    parser.add_argument("--quiet", action="store_true", help="Only print the model answer (suppress retrieval info)")
    parser.add_argument("--show-chunks", action="store_true", help="Also print the retrieved chunks after the answer")
    parser.add_argument("--k", type=int, default=5, help="top-k chunks to retrieve")
    parser.add_argument(
        "--structured-mode",
        type=str,
        choices=["raw", "llm"],
        default="llm",
        help="How to render DB-structured answers: raw template or OpenAI-refined",
    )
    args = parser.parse_args()

    # Reduce noisy logs when quiet mode is requested
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
        for logger_name in ("httpx", "sentence_transformers", "huggingface_hub", "transformers", "retrieval"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rag_root = Path(__file__).resolve().parents[1]
    retrieval = load_retrieval_module(rag_root)
    structured = load_structured_retrieval_module(rag_root)

    db_path = (rag_root / args.db_path).resolve() if not Path(args.db_path).is_absolute() else Path(args.db_path)

    loaded_retrieval = False
    meta = None
    faiss_idx = None
    embeddings = None
    model = None

    def ensure_retrieval_loaded():
        nonlocal loaded_retrieval, meta, faiss_idx, embeddings, model
        if loaded_retrieval:
            return

        out_dir = Path(args.out)
        meta = retrieval.load_metadata(out_dir / retrieval.METADATA_FILENAME)
        faiss_idx = None
        embeddings = None
        if getattr(retrieval, '_HAS_FAISS', False) and (out_dir / retrieval.FAISS_INDEX_FILENAME).exists():
            faiss_idx = retrieval.load_faiss_index(out_dir / retrieval.FAISS_INDEX_FILENAME)
        else:
            embeddings = retrieval.load_embeddings_np(out_dir / retrieval.EMBEDDINGS_FILENAME)

        print("Loading embedding model (only once)...")
        model = retrieval.load_embedding_model()
        print("Ready. Retrieval loaded.")
        loaded_retrieval = True

    def handle_query(q: str):
        if structured.should_route_to_courses_db(q):
            course_answer = structured.answer_course_query(q, db_path)
            if course_answer is not None:
                final_answer = render_structured_answer(q, course_answer, args)
                print()
                print(f'Απάντηση: "{final_answer}"')
                if args.show_chunks and not args.quiet:
                    print("DB source:")
                    print(
                        f"table={course_answer.get('source_table')} "
                        f"course_id={course_answer.get('course_id')} "
                        f"intent={course_answer.get('intent')} "
                        f"score={course_answer.get('score'):.2f}"
                    )
                return

        if structured.should_route_to_staff_db(q):
            structured_answer = structured.answer_staff_query(q, db_path)
            if structured_answer is not None:
                final_answer = render_structured_answer(q, structured_answer, args)
                print()
                print(f'Απάντηση: "{final_answer}"')
                if args.show_chunks and not args.quiet:
                    print("DB source:")
                    print(
                        f"table={structured_answer.get('source_table')} "
                        f"staff_id={structured_answer.get('staff_id')} "
                        f"score={structured_answer.get('score'):.2f}"
                    )
                return

        ensure_retrieval_loaded()
        t0 = time()
        res = retrieval.retrieve(q, model, faiss_idx, embeddings, meta, top_k=args.k)
        dt = time() - t0
        if not args.quiet:
            print(f"Retrieved {len(res['results'])} chunks in {dt:.2f}s")
        context = res.get("context", "")
        prompt = build_prompt_from_context(context, q)
        try:
            api_key = os.environ.get(args.api_key_env, "").strip()
            if not api_key:
                print(f"Missing API key in env var: {args.api_key_env}")
                return
            answer = openai_chat(args.api_base_url, api_key, args.model, prompt)
        except Exception as e:
            print("OpenAI call failed:", e)
            return
        # print the answer clearly: blank line + label
        clean_answer = answer.strip()
        print()
        print(f'Απάντηση: "{clean_answer}"')

        # Optionally print the individual retrieved chunks
        if args.show_chunks:
            results = res.get('results', [])
            def _format_chunk(item, idx):
                # extract common fields from different retrieval shapes
                score = item.get('score') if isinstance(item, dict) else None
                ridx = item.get('index') if isinstance(item, dict) else None
                meta = item.get('meta', {}) if isinstance(item, dict) else {}
                source = item.get('source') or meta.get('source') or meta.get('filename')
                section = meta.get('section') or item.get('section')

                # get the best available text field
                text = None
                for k in ('retrieval_text', 'text', 'content', 'chunk'):
                    if isinstance(item, dict) and k in item and isinstance(item[k], str):
                        text = item[k]
                        break
                if not text and isinstance(meta, dict):
                    text = meta.get('text') or meta.get('content')

                if text is None:
                    text = json.dumps(item, ensure_ascii=False)

                # pretty format
                header = f"Chunk {idx}"
                lines = []
                lines.append("" + "=" * 60)
                lines.append(f"{header}  (retrieval_index={ridx}  score={score})")
                if source:
                    lines.append(f"Source: {source}")
                if section:
                    lines.append(f"Section: {section}")
                lines.append("")
                # wrap text for console width
                wrapped = textwrap.fill(text.strip(), width=100)
                lines.append(wrapped)
                lines.append("" + "=" * 60)
                return "\n".join(lines)

            if not args.quiet:
                print("Retrieved chunks:")
            for i, it in enumerate(results, start=1):
                print(_format_chunk(it, i))

    if args.query:
        handle_query(args.query)
        return

    print("RAG CLI interactive. Type your question or Ctrl-C to quit.")
    try:
        while True:
            q = input("You: ")
            if not q.strip():
                continue
            handle_query(q.strip())
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == '__main__':
    main()
