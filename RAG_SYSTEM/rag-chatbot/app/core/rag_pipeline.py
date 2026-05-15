from typing import List, Any
from app.nlp.preprocess import clean_text
from app.retriever.vectorstore import VectorStore

class RAGPipeline:
    def __init__(self, vectorstore: VectorStore, retriever_top_k: int = 5):
        self.vectorstore = vectorstore
        self.retriever_top_k = retriever_top_k

    def retrieve(self, query: str) -> List[Any]:
        q = clean_text(query)
        # In a complete pipeline you'd encode `q` and pass an embedding to the vectorstore.
        return self.vectorstore.search(q, top_k=self.retriever_top_k)

    def generate_prompt(self, query: str, docs: List[str]) -> str:
        from app.core.prompt_builder import build_prompt
        return build_prompt(query, docs)

    def run(self, query: str) -> dict:
        docs = self.retrieve(query)
        prompt = self.generate_prompt(query, docs)
        return {"query": query, "docs": docs, "prompt": prompt}
