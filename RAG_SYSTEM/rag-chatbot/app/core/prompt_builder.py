from typing import List

def build_prompt(query: str, docs: List[str]) -> str:
    context = "\n\n".join(docs) if docs else ""
    return f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\n"
