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
import contextlib
import io
from typing import Any, Dict, List, Optional


NO_EVIDENCE_ANSWER = "Δεν βρέθηκε σχετική πληροφορία."
STRICT_RAG_SYSTEM_PROMPT = (
    "Απάντησε ΜΟΝΟ χρησιμοποιώντας τα αποσπάσματα που δίνονται. "
    "Αν η απάντηση δεν υπάρχει στα αποσπάσματα, απάντησε ακριβώς: \"Δεν βρέθηκε σχετική πληροφορία.\" "
    "ΜΗΝ χρησιμοποιείς εξωτερική γνώση."
)


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
    retrieval_dir = str(retrieval_path.parent.resolve())
    if retrieval_dir not in sys.path:
        sys.path.insert(0, retrieval_dir)
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


def openai_chat(api_base_url: str,
                api_key: str,
                model: str,
                prompt: str,
                temperature: float = 0.0,
                system_prompt: Optional[str] = None):
    url = api_base_url.rstrip("/") + "/chat/completions"
    sys_prompt = system_prompt or "You are a helpful assistant. Use the provided context to answer precisely."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
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
    return (
        "### CONTEXT\n"
        f"{context}\n\n"
        "### QUESTION\n"
        f"{query}\n\n"
        "### ANSWER\n"
    )


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


class RAGEngine:
    def __init__(
        self,
        out: str = "../Embedding/out",
        db_path: str = "../IEE_IHU_DB/NAO_IEE_IHU_DB.db",
        api_base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4o-mini",
        quiet: bool = True,
        k: int = 5,
        structured_mode: str = "llm",
        preload_retrieval_on_boot: bool = True,
    ) -> None:
        self.out = out
        self.db_path_arg = db_path
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.model_name = model
        self.quiet = quiet
        self.k = k
        self.structured_mode = structured_mode
        self.preload_retrieval_on_boot = preload_retrieval_on_boot

        _load_dotenv_from_parents(Path(__file__).resolve().parent)

        if self.quiet:
            logging.basicConfig(level=logging.WARNING)
            for logger_name in ("httpx", "sentence_transformers", "huggingface_hub", "transformers", "retrieval"):
                logging.getLogger(logger_name).setLevel(logging.WARNING)
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.rag_root = Path(__file__).resolve().parents[1]
        self.retrieval = load_retrieval_module(self.rag_root)
        self.structured = load_structured_retrieval_module(self.rag_root)

        self.generator_dir = Path(__file__).resolve().parent
        self.db_path = self._resolve_path(self.db_path_arg)

        self.loaded_retrieval = False
        self.meta = None
        self.faiss_idx = None
        self.embeddings = None
        self.bm25_idx = None
        self.model = None

        if self.preload_retrieval_on_boot:
            try:
                self._ensure_retrieval_loaded()
            except Exception as e:
                print(f"[WARN] Retrieval preload on boot failed: {e}")

    def _resolve_path(self, raw_path: str) -> Path:
        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            return path_obj

        # Prefer an existing target from common execution roots.
        candidates = [
            (Path.cwd() / path_obj).resolve(),
            (self.generator_dir / path_obj).resolve(),
            (self.rag_root / path_obj).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Keep backward-compatible fallback (historically anchored at rag_root).
        return candidates[-1]

    def _ensure_retrieval_loaded(self) -> None:
        if self.loaded_retrieval:
            return

        out_dir = self._resolve_path(self.out)

        self.meta = self.retrieval.load_metadata(out_dir / self.retrieval.METADATA_FILENAME)
        self.faiss_idx = None
        self.embeddings = self.retrieval.load_embeddings_np(out_dir / self.retrieval.EMBEDDINGS_FILENAME)
        if getattr(self.retrieval, "_HAS_FAISS", False) and (out_dir / self.retrieval.FAISS_INDEX_FILENAME).exists():
            self.faiss_idx = self.retrieval.load_faiss_index(out_dir / self.retrieval.FAISS_INDEX_FILENAME)

        if self.faiss_idx is None and self.embeddings is None:
            raise RuntimeError("No retrieval index available: both FAISS and embeddings are missing")

        # Build/load sparse index from the same metadata corpus used for dense retrieval.
        try:
            self.bm25_idx = self.retrieval.load_or_build_bm25(out_dir, self.meta)
        except Exception as e:
            print(f"[WARN] BM25 index unavailable, continuing dense-only retrieval: {e}")
            self.bm25_idx = None

        print("Loading embedding model (only once)...")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = self.retrieval.load_embedding_model()
        print("Ready. Retrieval loaded.")
        self.loaded_retrieval = True

    def _render_structured_answer(self, query: str, structured_answer: dict) -> str:
        if self.structured_mode == "raw":
            return (structured_answer.get("answer") or "").strip()

        prompt = build_prompt_from_structured_facts(query, structured_answer)
        try:
            api_key = os.environ.get(self.api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(f"Missing API key in env var: {self.api_key_env}")
            llm_answer = openai_chat(
                self.api_base_url,
                api_key,
                self.model_name,
                prompt,
                temperature=0.2,
            ).strip()
            if llm_answer:
                return llm_answer
        except Exception:
            pass

        return (structured_answer.get("answer") or "").strip()

    def answer_query(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"answer": None, "chunks": [], "error": "Empty query", "route": "none"}

        if self.structured.should_route_to_courses_db(q):
            course_answer = self.structured.answer_course_query(q, self.db_path)
            if course_answer is not None:
                return {
                    "answer": self._render_structured_answer(q, course_answer),
                    "chunks": [],
                    "error": None,
                    "route": "courses_db",
                    "source": course_answer,
                }

        if self.structured.should_route_to_staff_db(q):
            staff_answer = self.structured.answer_staff_query(q, self.db_path)
            if staff_answer is not None:
                return {
                    "answer": self._render_structured_answer(q, staff_answer),
                    "chunks": [],
                    "error": None,
                    "route": "staff_db",
                    "source": staff_answer,
                }

        self._ensure_retrieval_loaded()
        _t0 = time()
        res = self.retrieval.retrieve(
            q,
            self.model,
            self.faiss_idx,
            self.embeddings,
            self.meta,
            bm25_index=self.bm25_idx,
            top_k=self.k,
            dense_top_n=20,
            sparse_top_n=20,
            rrf_k=60,
            min_chunks_for_context=5,
            max_chunks_for_context=10,
            context_max_chars=12000,
        )
        _ = time() - _t0

        context = res.get("context", "")
        if not res.get("results"):
            return {
                "answer": NO_EVIDENCE_ANSWER,
                "chunks": [],
                "error": None,
                "route": "retrieval",
                "query_analysis": res.get("query_analysis", {}),
            }

        prompt = build_prompt_from_context(context, q)
        try:
            api_key = os.environ.get(self.api_key_env, "").strip()
            if not api_key:
                return {
                    "answer": None,
                    "chunks": res.get("results", []),
                    "error": f"Missing API key in env var: {self.api_key_env}",
                    "route": "retrieval",
                }
            answer = openai_chat(
                self.api_base_url,
                api_key,
                self.model_name,
                prompt,
                temperature=0.0,
                system_prompt=STRICT_RAG_SYSTEM_PROMPT,
            ).strip()
            if not answer:
                answer = NO_EVIDENCE_ANSWER
        except Exception as e:
            return {
                "answer": None,
                "chunks": res.get("results", []),
                "error": f"OpenAI call failed: {e}",
                "route": "retrieval",
            }

        return {
            "answer": answer,
            "chunks": res.get("results", []),
            "error": None,
            "route": "retrieval",
            "query_analysis": res.get("query_analysis", {}),
            "retrieval_debug": {
                "requested_top_k": res.get("requested_top_k"),
                "final_result_count": res.get("final_result_count"),
                "context_chunk_count": res.get("context_chunk_count"),
                "context_truncated": res.get("context_truncated"),
                "context_char_len": res.get("context_char_len"),
                "filtered_corpus_size": res.get("filtered_corpus_size"),
            },
        }


def format_chunk(item: dict, idx: int) -> str:
    score = item.get("score") if isinstance(item, dict) else None
    ridx = item.get("index") if isinstance(item, dict) else None
    meta = item.get("meta", {}) if isinstance(item, dict) else {}
    source = item.get("source") or meta.get("source") or meta.get("filename")
    section = meta.get("section") or item.get("section")

    text = None
    for key in ("retrieval_text", "text", "content", "chunk"):
        if isinstance(item, dict) and key in item and isinstance(item[key], str):
            text = item[key]
            break
    if not text and isinstance(meta, dict):
        text = meta.get("text") or meta.get("content")

    if text is None:
        text = json.dumps(item, ensure_ascii=False)

    lines: List[str] = []
    lines.append("" + "=" * 60)
    lines.append(f"Chunk {idx}  (retrieval_index={ridx}  score={score})")
    if source:
        lines.append(f"Source: {source}")
    if section:
        lines.append(f"Section: {section}")
    lines.append("")
    lines.append(textwrap.fill(text.strip(), width=100))
    lines.append("" + "=" * 60)
    return "\n".join(lines)


def main():
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

    engine = RAGEngine(
        out=args.out,
        db_path=args.db_path,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        quiet=args.quiet,
        k=args.k,
        structured_mode=args.structured_mode,
    )

    def handle_query(q: str):
        result = engine.answer_query(q)
        clean_answer = (result.get("answer") or "").strip()
        if not clean_answer:
            err = result.get("error") or "No answer"
            print(err)
            return

        print()
        print(f'Απάντηση: "{clean_answer}"')

        if not args.quiet:
            debug = result.get("retrieval_debug") or {}
            if debug:
                print(
                    "Retrieval debug: "
                    f"requested_top_k={debug.get('requested_top_k')} "
                    f"selected_chunks={debug.get('final_result_count')} "
                    f"sent_to_llm={debug.get('context_chunk_count')} "
                    f"context_chars={debug.get('context_char_len')} "
                    f"truncated={debug.get('context_truncated')}"
                )

        # Optionally print the individual retrieved chunks
        if args.show_chunks:
            results = result.get("chunks", [])

            if not args.quiet:
                print("Retrieved chunks:")
            for i, it in enumerate(results, start=1):
                print(format_chunk(it, i))

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
