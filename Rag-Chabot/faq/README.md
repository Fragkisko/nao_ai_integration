# Semantic FAQ Layer

Αυτός ο φάκελος υλοποιεί ανεξάρτητο semantic FAQ layer που εκτελείται πριν από το κανονικό RAG retrieval.

## Αρχεία

- `faq.json`: authoritative FAQ entries (question/answer/category)
- `build_faq_index.py`: index builder για semantic FAQ
- `semantic_faq.py`: runtime semantic FAQ searcher
- `embeddings.npy`, `metadata.json`, `faiss.index`: παραγόμενα artifacts του FAQ index

## Pipeline προτεραιότητας

1. Structured Retrieval
2. Semantic FAQ
3. RAG Retrieval (BM25 + FAISS + Hybrid + LLM)

Αν υπάρξει structured answer ή confident FAQ match, η ροή σταματάει πριν το κανονικό RAG.

## Build FAQ index

```powershell
python Rag-Chabot/faq/build_faq_index.py --faq-json Rag-Chabot/faq/faq.json --out Rag-Chabot/faq
```

## Config (env vars)

- `FAQ_THRESHOLD` (default `0.88`): semantic similarity threshold
- `ENABLE_FAQ_REPHRASE` (default `false`): αν `true`, γίνεται προαιρετικό LLM rephrase της FAQ απάντησης

## FAQ JSON format

```json
[
  {
    "id": 1,
    "question": "Πού βρίσκεται η γραμματεία;",
    "answer": "Η γραμματεία βρίσκεται ...",
    "category": "Administration"
  }
]
```
