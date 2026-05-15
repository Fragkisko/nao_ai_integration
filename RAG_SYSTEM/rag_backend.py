"""RAG backend skeleton helpers.

Αυτό το αρχείο παρέχει κεντρικές helper συναρτήσεις (stubs) για το RAG
pipeline: ταξινόμηση query, DB lookup, ανάκτηση chunks, reranking,
συναρμολόγηση prompt και κλήση στο Ollama. Είναι skeleton — κάθε
συνάρτηση πρέπει να συμπληρωθεί από εσένα ώστε να ταιριάξει με τα
εργαλεία/DB/embedding models που επιλέξεις.

Μη δρομολογήσεις αυτόματα αυτό το module — είναι προς συμπλήρωση.
"""
from typing import List, Dict, Any, Optional


def classify_query(text: str) -> Dict[str, Any]:
    """Normalize και εξάγει intent + entities από το user query.

    Επιστρέφει dict, παράδειγμα:
    {
        "normalized": "πες μου για την πρακτική άσκηση",
        "intent": "program_info",
        "entities": {"program": "πρακτική άσκηση"},
        "confidence": 0.9,
    }

    TODO: συμπλήρωσε με τον classifier/NER που προτιμάς.
    """
    raise NotImplementedError("implement classify_query")


def db_lookup(intent: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Κάνει fast path lookup στη δομημένη DB `NAO_IEE_IHU_DB.db`.

    Επιστρέφει record dict ή None αν δεν υπάρχει match.
    Το record πρέπει να περιέχει τουλάχιστον τα πεδία: `short_answer`, `full_answer`, `source`.
    """
    raise NotImplementedError("implement db_lookup")


def retrieve_chunks(query: str, *, k: int = 5) -> List[Dict[str, Any]]:
    """Ανακτά chunks από το vector index βάσει embedding του query.

    Κάθε chunk dict πρέπει να περιέχει: `text`, `source_name`, `path_or_id`, `score`, `metadata`.
    """
    raise NotImplementedError("implement retrieve_chunks")


def rerank(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Εφαρμόζει reranking recipe και επιστρέφει ταξινομημένα chunks.

    Μπορεί να χρησιμοποιήσει συνδυασμό similarity, keyword overlap, metadata match.
    """
    raise NotImplementedError("implement rerank")


def build_prompt(query: str, chunks: List[Dict[str, Any]], *, preferred_output_length: str = "short") -> Dict[str, Any]:
    """Συναρμολογεί το system+user prompt payload για το Ollama.

    Επιστρέφει dict με πεδία όπως: `system_message`, `user_message`, `messages` (για chat API).
    Το format θα ταιριάξει με τη συνάρτηση `call_ollama` που θα χρησιμοποιήσεις.
    """
    raise NotImplementedError("implement build_prompt")


def call_ollama(system_message: str, user_message: str, *, model: Optional[str] = None, timeout_sec: int = 120, options: Optional[Dict[str, Any]] = None) -> str:
    """Καλεί το Ollama (ή άλλο local LLM) με το δοθέν payload και επιστρέφει το string απάντηση.

    Αυτό είναι wrapper — μπορείς να χρησιμοποιήσεις `App/nao_common.py:ollama_generate` ή το chat endpoint.
    """
    raise NotImplementedError("implement call_ollama")


def assemble_answer(db_record: Optional[Dict[str, Any]], chunks: List[Dict[str, Any]], llm_reply: Optional[str], *, preferred_output_length: str = "short") -> Dict[str, Any]:
    """Συνδυάζει DB summary / chunks / llm_reply σε τελικό response dict.

    Επιστρέφει: {
        "answer": str,
        "used_sources": [ {"source_name":..., "path_or_id":..., "confidence":...}, ... ],
        "confidence": float
    }
    """
    raise NotImplementedError("implement assemble_answer")
