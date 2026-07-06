import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


KNOWLEDGE_BASE_LABELS = {
    "bachelor": "Undergraduate (Bachelor)",
    "master": "Postgraduate (Master + PhD)",
}


def normalize_knowledge_base_mode(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    aliases = {
        "1": "bachelor",
        "bachelor": "bachelor",
        "undergraduate": "bachelor",
        "2": "master",
        "master": "master",
        "postgraduate": "master",
        "phd": "master",
    }
    return aliases.get(value)


def prompt_for_knowledge_base_mode() -> str:
    while True:
        print("Select knowledge base:\n")
        print("1 -> Undergraduate (Bachelor)")
        print("2 -> Postgraduate (Master + PhD)")
        choice = input("Selection: ").strip()
        mode = normalize_knowledge_base_mode(choice)
        if mode:
            return mode
        print("Invalid selection. Please choose 1 or 2.\n")


def resolve_knowledge_base_out_dir(mode: str) -> Path:
    normalized_mode = normalize_knowledge_base_mode(mode)
    if normalized_mode is None:
        raise ValueError(f"Unsupported knowledge base mode: {mode}")

    return Path(__file__).resolve().parents[1] / "Rag-Chabot" / "knowledge_base" / normalized_mode


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
    knowledge_base_mode: str = "bachelor",
) -> Tuple[Optional[Any], Optional[Any]]:
    rag_module = None
    rag_engine = None
    try:
        rag_module = load_rag_cli_module()
        rag_engine = rag_module.RAGEngine(
            out=str(resolve_knowledge_base_out_dir(knowledge_base_mode)),
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
