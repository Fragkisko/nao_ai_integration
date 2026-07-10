from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Rag-Chabot" / "Retrival" / "structured_retrieval.py"
DB_PATH = ROOT / "IEE_IHU_DB" / "NAO_IEE_IHU_DB.db"


def _load_structured_staff_module():
    spec = importlib.util.spec_from_file_location("structured_staff_runtime", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load structured_retrieval module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_staff_entity_route_without_professor_keyword():
    sr = _load_structured_staff_module()
    query = "τι γνωριζεις για τον κεραμοπουλος ευκλειδης"

    assert sr.should_route_to_staff_db(query) is True


def test_staff_entity_answer_without_professor_keyword():
    sr = _load_structured_staff_module()
    query = "τι γνωριζεις για τον κεραμοπουλος ευκλειδης"

    result = sr.answer_staff_query(query, DB_PATH)
    assert result is not None
    full_name = sr.normalize_text(str((result.get("row") or {}).get("full_name") or ""))
    assert "κεραμοπ" in full_name


def test_staff_last_name_only_query():
    sr = _load_structured_staff_module()
    query = "τι γνωριζεις για κεραμοπουλος"

    assert sr.should_route_to_staff_db(query) is True
    result = sr.answer_staff_query(query, DB_PATH)
    assert result is not None
    full_name = sr.normalize_text(str((result.get("row") or {}).get("full_name") or ""))
    assert "κεραμοπ" in full_name


def test_non_staff_concept_query_not_routed_to_staff():
    sr = _load_structured_staff_module()
    query = "εξηγησε μου τι ειναι νευρωνικα δικτυα"

    assert sr.should_route_to_staff_db(query) is False


def test_staff_count_query_answered_structured():
    sr = _load_structured_staff_module()
    query = "Πόσοι καθηγητές υπάρχουν στο τμήμα και πόσοι στο ΔΙΠΑΕ γενικά"

    assert sr.should_route_to_staff_db(query) is True
    result = sr.answer_staff_query(query, DB_PATH)
    assert result is not None
    assert result.get("operation") == "count_staff"
    assert int(result.get("total_count") or 0) > 0
