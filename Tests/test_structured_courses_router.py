from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Rag-Chabot" / "Retrival" / "structured_courses.py"
DB_PATH = ROOT / "IEE_IHU_DB" / "NAO_IEE_IHU_DB.db"


def _load_structured_courses_module():
    spec = importlib.util.spec_from_file_location("structured_courses_runtime", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load structured_courses module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_structured_course_entity_lookup():
    sc = _load_structured_courses_module()
    result = sc.answer_course_query_structured("τι γνωρίζεις για δομημένο προγραμματισμό", DB_PATH)

    assert result is not None
    assert result.get("operation") == "course_entity_lookup"
    row = result.get("payload", {}).get("row", {})
    title = sc.normalize_text(str(row.get("title") or ""))
    assert "δομημενος προγραμματισμος" in title


def test_rag_fallback_for_non_course_topic():
    sc = _load_structured_courses_module()
    result = sc.answer_course_query_structured("εξήγησε μου τι είναι νευρωνικά δίκτυα", DB_PATH)

    # Router should not force structured answer for non-course conceptual topics.
    if result is None:
        assert True
        return

    assert bool(result.get("should_fallback_to_rag")) is True


def test_aggregation_count_courses():
    sc = _load_structured_courses_module()
    result = sc.answer_course_query_structured("πόσα μαθήματα υπάρχουν", DB_PATH)

    assert result is not None
    assert result.get("operation") == "count_courses"
    total = int(result.get("payload", {}).get("total_count") or 0)
    assert total > 0
