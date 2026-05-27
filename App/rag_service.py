import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_rag_cli_module():
    rag_cli_path = Path(__file__).resolve().parents[1] / "Rag-Chabot" / "Generator" / "rag_cli.py"
    spec = importlib.util.spec_from_file_location("rag_cli_runtime", str(rag_cli_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rag_cli module from {rag_cli_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_rag_engine(
    model: str,
    top_k: int,
    preload_retrieval_on_boot: bool = True,
) -> Tuple[Optional[Any], Optional[Any]]:
    rag_module = None
    rag_engine = None
    try:
        rag_module = load_rag_cli_module()
        rag_engine = rag_module.RAGEngine(
            out="../Embedding/out",
            model=model,
            quiet=True,
            k=top_k,
            structured_mode="llm",
            preload_retrieval_on_boot=preload_retrieval_on_boot,
        )
    except Exception as e:
        print(f"[WARN] RAG init error: {e}")
    return rag_module, rag_engine


def answer_query(rag_engine: Optional[Any], user_text: str) -> Dict[str, Any]:
    if rag_engine is None:
        return {"answer": None, "error": "RAG engine unavailable", "chunks": []}

    try:
        return rag_engine.answer_query(user_text)
    except Exception as e:
        return {"answer": None, "error": f"RAG runtime error: {e}", "chunks": []}
