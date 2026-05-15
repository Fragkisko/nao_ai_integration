class QueryClassifier:
    """Lightweight placeholder classifier used by the RAG app."""

    def __init__(self):
        pass

    def classify(self, text: str) -> str:
        text = text or ""
        lower = text.strip().lower()
        if "?" in text or lower.startswith(("how", "what", "why", "when")):
            return "question"
        if "error" in lower or "bug" in lower:
            return "technical"
        return "general"
