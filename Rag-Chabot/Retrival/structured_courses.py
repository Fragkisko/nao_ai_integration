"""Structured course retrieval with operation-based planning and safe SQL.

This module handles course-focused structured questions directly from SQLite.
It is independent from embedding retrieval and uses only parameterized SQL.
"""

from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple


DB_TABLE_COURSES = "iee_ihu_courses"
DEFAULT_LIST_LIMIT = 7
COURSE_ENTITY_MATCH_MIN_SCORE = 0.62


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = _strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text


def _decode_sqlite_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        for enc in ("utf-8", "cp1253", "latin-1"):
            try:
                return value.decode(enc)
            except Exception:
                continue
        return value.decode("utf-8", errors="replace")
    return value


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in row.keys():
        out[key] = _decode_sqlite_text(row[key])
    return out


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.text_factory = bytes
    conn.row_factory = sqlite3.Row
    return conn


def _canonicalize_filter_value(field: str, raw_value: str) -> str:
    value = normalize_text(raw_value)
    # Synonym normalization into canonical DB values (Greek labels in this dataset).
    synonym_map = {
        "course_type": {
            "general infrastructure": "υποδομ",
            "specialized infrastructure": "υποδομ",
            "scientific area": "επιστημον",
            "specialization": "εξειδικ",
            "skills development": "δεξιοτητ",
            "γενικης υποδομης": "υποδομ",
            "ειδικης υποδομης": "υποδομ",
            "επιστημονικης περιοχης": "επιστημον",
            "εξειδικευσης": "εξειδικ",
            "αναπτυξης δεξιοτητων": "δεξιοτητ",
        },
        "course_kind": {
            "mandatory": "υποχρεωτ",
            "required": "υποχρεωτ",
            "compulsory": "υποχρεωτ",
            "υποχρεωτικο": "υποχρεωτ",
            "elective": "επιλογ",
            "optional": "προαιρετ",
            "choice": "επιλογ",
            "υποχρεωτικο επιλογης": "επιλογ",
            "υποχρεωτικο επιλογης (υπ-επ)": "επιλογ",
            "επιλογης": "επιλογ",
            "προαιρετικο": "προαιρετ",
        },
    }

    mapped = synonym_map.get(field, {}).get(value)
    return mapped or value


def _extract_semester(query_norm: str) -> Optional[str]:
    m = re.search(r"(?:semester|sem|εξαμην(?:ο|ου|α)|term)\s*(\d{1,2})", query_norm)
    if m:
        return m.group(1)

    m2 = re.search(r"\b(\d{1,2})(?:ο|ου|ος|η|th|st|nd|rd)?\s*(?:semester|sem|εξαμην(?:ο|ου|α))", query_norm)
    if m2:
        return m2.group(1)

    m3 = re.search(r"\b(\d{1,2})(?:ο|ου|ος|η)\b", query_norm)
    if m3 and "εξαμην" in query_norm:
        return m3.group(1)

    return None


def _extract_course_code(query: str) -> Optional[str]:
    m = re.search(r"\b(\d{4})\b", query or "")
    return m.group(1) if m else None


def _extract_named_group(query_norm: str, patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, query_norm)
        if m:
            value = (m.group(1) or "").strip(" .,:;!?\")('[]{}")
            if value:
                return value
    return None


def _tokenize_text(text: str) -> List[str]:
    raw = re.findall(r"[\w\u0370-\u03ff]+", normalize_text(text or ""), flags=re.UNICODE)
    stop = {
        "τι",
        "τις",
        "τις",
        "το",
        "τον",
        "την",
        "τους",
        "των",
        "για",
        "και",
        "μου",
        "πες",
        "γνωριζεις",
        "γνωριζω",
        "πληροφοριες",
        "πληροφορια",
        "θελω",
        "μαθημα",
        "μαθηματος",
        "μαθηματα",
        "ειναι",
        "ειναι",
    }
    return [t for t in raw if len(t) >= 3 and t not in stop]


def _stem_token(token: str) -> str:
    t = normalize_text(token)
    for suffix in ("ων", "ους", "ου", "ος", "ης", "η", "α", "ο", "ες", "ς"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def _token_stems(text: str) -> List[str]:
    return [_stem_token(t) for t in _tokenize_text(text)]


def _extract_course_lookup_candidate(query: str) -> str:
    q = normalize_text(query)
    patterns = (
        r"^τι\s+γνωριζεις\s+για\s+(?:το\s+)?(?:μαθημα\s+)?(.+)$",
        r"^πες\s+μου\s+πληροφοριες\s+για\s+(?:το\s+)?(?:μαθημα\s+)?(.+)$",
        r"^θελω\s+πληροφοριες\s+για\s+(?:το\s+)?(?:μαθημα\s+)?(.+)$",
        r"^τι\s+περιλαμβανει\s+(?:το\s+)?(?:μαθημα\s+)?(.+)$",
        r"^τι\s+ειναι\s+(?:το\s+)?(?:μαθημα\s+)?(.+)$",
        r"^ποιο\s+ειναι\s+το\s+μαθημα\s+(.+)$",
    )
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return (m.group(1) or "").strip(" .,:;!?()[]{}\"")
    return q.strip(" .,:;!?()[]{}\"")


def _entity_match_score(candidate: str, row: Dict[str, Any]) -> Tuple[float, str]:
    cand = normalize_text(candidate)
    title = normalize_text(str(row.get("title") or ""))
    code = normalize_text(str(row.get("course_code") or ""))

    if not cand or not title:
        return 0.0, "none"

    if cand == title:
        return 1.0, "exact"
    if cand == code:
        return 0.99, "exact_code"

    if cand in title or title in cand:
        return 0.92, "contains"

    cand_tokens = set(_token_stems(cand))
    title_tokens = set(_token_stems(title))
    if not cand_tokens or not title_tokens:
        seq = SequenceMatcher(None, cand, title).ratio()
        return seq * 0.8, "fuzzy"

    overlap = len(cand_tokens & title_tokens) / max(1, len(cand_tokens))
    seq = SequenceMatcher(None, cand, title).ratio()
    score = (0.65 * overlap) + (0.35 * seq)
    if cand_tokens.issubset(title_tokens):
        score += 0.08
    if code and code in cand:
        score += 0.1
    return min(score, 0.98), "fuzzy"


def find_best_course_entity_match(query: str, db_path: Path) -> Optional[Dict[str, Any]]:
    candidate = _extract_course_lookup_candidate(query)
    rows = _fetch_all_course_rows(db_path)
    if not candidate or not rows:
        return None

    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for row in rows:
        score, match_type = _entity_match_score(candidate, row)
        if score > 0:
            ranked.append((score, match_type, row))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0], reverse=True)
    score, match_type, row = ranked[0]
    return {
        "score": score,
        "match_type": match_type,
        "candidate": candidate,
        "row": row,
    }


def _build_course_entity_answer(row: Dict[str, Any]) -> str:
    code = str(row.get("course_code") or "-").strip()
    title = str(row.get("title") or "Άγνωστος τίτλος").strip()
    semester = str(row.get("semester") or "-").strip()
    ects = row.get("ects")
    course_type = str(row.get("course_type") or "").strip()
    course_kind = str(row.get("course_kind") or "").strip()
    knowledge_area = str(row.get("knowledge_area") or "").strip()
    weekly_hours = str(row.get("weekly_hours") or "").strip()
    coordinator = str(row.get("coordinator") or "").strip()
    content = str(row.get("content") or "").strip()
    course_url = str(row.get("course_url") or "").strip()

    parts = [f"Το μάθημα {code} είναι το «{title}» (εξάμηνο: {semester}, ECTS: {ects})."]
    if course_type:
        parts.append(f"Τύπος μαθήματος: {course_type}.")
    if course_kind:
        parts.append(f"Κατηγορία: {course_kind}.")
    if knowledge_area:
        parts.append(f"Γνωστικό αντικείμενο: {knowledge_area}.")
    if weekly_hours:
        parts.append(f"Εβδομαδιαίες ώρες: {weekly_hours}.")
    if coordinator:
        parts.append(f"Συντονιστής: {coordinator}.")
    if content:
        parts.append(f"Περιεχόμενο: {content}.")
    if course_url:
        parts.append(f"Περισσότερα: {course_url}")
    return " ".join(parts)


def _is_semester_like_topic(topic: str) -> bool:
    t = normalize_text(topic)
    if "εξαμην" in t or "semester" in t or "sem" in t:
        return True
    if re.search(r"\b\d{1,2}(?:ο|ου|ος|η)?\b", t):
        return True
    return False


def _is_generic_topic_noise(topic: str) -> bool:
    t = normalize_text(topic)
    noise_terms = (
        "της σχολης",
        "στη σχολη",
        "σχολης",
        "σχολη",
        "του τμηματος",
        "του τμημα",
        "τμηματος",
        "τμημα",
        "ολα",
        "ολα",
        "υπαρχουν",
        "υπάρχουν",
        "ειναι",
        "είναι",
    )
    return any(term in t for term in noise_terms)


def _extract_filters(query: str) -> Tuple[Dict[str, str], bool]:
    q = normalize_text(query)
    filters: Dict[str, str] = {}
    topic_only_hint = False

    semester = _extract_semester(q)
    if semester:
        filters["semester"] = semester

    code = _extract_course_code(query)
    if code:
        filters["course_code"] = code

    coordinator = _extract_named_group(
        q,
        (
            r"coordinated by\s+([\w\s\-\u0370-\u03ff]+)",
            r"coordinator\s*[:\-]?\s*([\w\s\-\u0370-\u03ff]+)",
            r"who teaches\s+([\w\s\-\u0370-\u03ff]+)",
            r"διδασκει\s+το\s+μαθημα\s+([\w\s\-\u0370-\u03ff]+)",
            r"συντονιστ(?:ης|η|ης)?\s*(?:του)?\s*([\w\s\-\u0370-\u03ff]+)",
        ),
    )
    if coordinator and not re.search(r"(?:what|which|ποιος|ποια)", coordinator):
        # For "who teaches X" this token usually refers to title, not coordinator.
        if "who teaches" in q or "διδασκει" in q:
            filters["title"] = coordinator
        else:
            filters["coordinator"] = coordinator

    knowledge_area = _extract_named_group(
        q,
        (
            r"knowledge\s+area\s+([\w\s\-\u0370-\u03ff]+)",
            r"γνωστικ(?:ο|ου)?\s+αντικειμεν(?:ο|ου)\s+([\w\s\-\u0370-\u03ff]+)",
            r"στο\s+πεδιο\s+([\w\s\-\u0370-\u03ff]+)",
        ),
    )
    if knowledge_area:
        filters["knowledge_area"] = knowledge_area

    if any(term in q for term in ("compulsory", "mandatory", "required", "υποχρεωτικ")):
        filters["course_kind"] = _canonicalize_filter_value("course_kind", "compulsory")
    if any(term in q for term in ("elective", "optional", "επιλογ")):
        filters["course_kind"] = _canonicalize_filter_value("course_kind", "elective")
    if any(term in q for term in ("laboratory", "lab", "εργαστηρ")):
        filters["laboratory"] = "1"

    topic = _extract_named_group(
        q,
        (
            r"show\s+all\s+([\w\s\-\u0370-\u03ff]+)\s+courses",
            r"list\s+all\s+([\w\s\-\u0370-\u03ff]+)\s+courses",
            r"μαθηματα\s+([\w\s\-\u0370-\u03ff]+)",
        ),
    )
    if (
        topic
        and "title" not in filters
        and "knowledge_area" not in filters
        and "course_kind" not in filters
        and "course_type" not in filters
        and not _is_semester_like_topic(topic)
        and not _is_generic_topic_noise(topic)
    ):
        filters["topic"] = topic
        topic_only_hint = True

    return filters, topic_only_hint


def _detect_operation(query: str) -> Optional[str]:
    q = normalize_text(query)
    if not q:
        return None

    course_context_markers = (
        "μαθημ",
        "course",
        "κωδικ",
        "κωδ",
        "ects",
        "συντονισ",
        "coordinator",
        "εξαμην",
        "semester",
    )

    # Program-level duration questions should not be interpreted as course listing/counting.
    if (
        ("διαρκεια" in q or "διάρκεια" in q)
        and ("σπουδ" in q or "study" in q)
    ) or (
        any(k in q for k in ("τυπικ", "τυπικα", "τυπικά"))
        and any(k in q for k in ("εξαμην", "semester", "σχολη", "σχολή", "school"))
    ):
        return "study_duration"

    if any(k in q for k in ("how many", "ποσα", "πόσα", "count", "πληθος", "πλήθος")):
        if not any(k in q for k in course_context_markers):
            return None
        if "ects" in q or "μοναδ" in q:
            return "sum_ects"
        if "per " in q or "by " in q or "ανα " in q or "ανά " in q or "group" in q:
            return "group_count"
        return "count_courses"

    if any(k in q for k in ("sum", "total ects", "συνολο ects", "συνολο μοναδων", "σύνολο μονάδων")):
        return "sum_ects"

    if any(k in q for k in ("distinct", "different", "μοναδικ", "τιμες", "τιμές")):
        return "distinct_values"

    if any(
        k in q
        for k in (
            "τι γνωριζ",
            "πληροφορι",
            "τι περιλαμβαν",
            "τι ειναι το μαθημα",
            "τι ειναι το",
            "θελω πληροφορι",
        )
    ):
        return "course_entity_lookup"

    list_markers = ("list", "show", "which", "δωσε", "δώσε")
    if any(k in q for k in list_markers):
        return "list_courses"

    # Greek "ποια" is ambiguous; treat as list only when course context exists.
    if ("ποια" in q or "ποιa" in q) and any(k in q for k in course_context_markers):
        return "list_courses"

    if any(k in q for k in ("coordinator", "who teaches", "διδασκ", "συντονιστ")):
        return "list_courses"

    return None


def _detect_group_field(query: str) -> str:
    q = normalize_text(query)
    if any(k in q for k in ("semester", "εξαμην")):
        return "semester"
    if any(k in q for k in ("knowledge area", "γνωστικ")):
        return "knowledge_area"
    if any(k in q for k in ("type", "τυπος", "τύπος")):
        return "course_type"
    if any(k in q for k in ("kind", "κατηγορια", "κατηγορία")):
        return "course_kind"
    if any(k in q for k in ("coordinator", "συντονιστ")):
        return "coordinator"
    return "semester"


def _study_duration_from_courses(db_path: Path) -> Dict[str, Any]:
    semesters = sorted(
        {
            int(_semester_digits(row.get("semester")))
            for row in _fetch_all_course_rows(db_path)
            if _semester_digits(row.get("semester"))
        }
    )
    total_semesters = len(semesters)
    max_semester = max(semesters) if semesters else 0

    # Prefer max semester as canonical duration when contiguous curriculum exists.
    effective_semesters = max_semester if max_semester > 0 else total_semesters
    duration_years = effective_semesters / 2 if effective_semesters > 0 else 0

    return {
        "total_semesters": total_semesters,
        "max_semester": max_semester,
        "effective_semesters": effective_semesters,
        "duration_years": duration_years,
    }


def _allowed_filter_fields() -> Dict[str, str]:
    return {
        "semester": "semester",
        "coordinator": "coordinator",
        "course_type": "course_type",
        "course_kind": "course_kind",
        "laboratory": "weekly_hours",
        "title": "title",
        "course_code": "course_code",
        "knowledge_area": "knowledge_area",
    }


def _build_where_clause(filters: Dict[str, str]) -> Tuple[str, List[Any]]:
    allowed = _allowed_filter_fields()
    clauses: List[str] = []
    params: List[Any] = []

    for key, raw_value in filters.items():
        if key == "topic":
            clauses.append("(LOWER(COALESCE(title, '')) LIKE ? OR LOWER(COALESCE(knowledge_area, '')) LIKE ?)")
            token = f"%{normalize_text(raw_value)}%"
            params.extend([token, token])
            continue

        field = allowed.get(key)
        if field is None:
            continue

        value = _canonicalize_filter_value(key, str(raw_value))
        if not value:
            continue

        if key == "semester":
            clauses.append("REPLACE(REPLACE(LOWER(COALESCE(CAST(semester AS TEXT), '')), 'ο', ''), 'o', '') = ?")
            params.append(str(value))
            continue

        if key == "laboratory":
            clauses.append("(LOWER(COALESCE(weekly_hours, '')) LIKE ? OR LOWER(COALESCE(weekly_hours, '')) LIKE ?)")
            params.append("%(ε)%")
            params.append("%/ 2 (ε)%")
            continue

        if key == "course_code":
            clauses.append("LOWER(COALESCE(course_code, '')) LIKE ?")
            params.append(f"%{normalize_text(value)}%")
            continue

        clauses.append(f"LOWER(COALESCE({field}, '')) LIKE ?")
        params.append(f"%{normalize_text(value)}%")

    if not clauses:
        return "", params

    return " WHERE " + " AND ".join(clauses), params


def _semester_digits(value: Any) -> str:
    text = normalize_text(str(value or ""))
    m = re.search(r"\d{1,2}", text)
    return m.group(0) if m else ""


def _contains_norm(haystack: Any, needle: Any) -> bool:
    h = normalize_text(str(haystack or ""))
    n = normalize_text(str(needle or ""))
    if not n:
        return True
    return n in h


def _fetch_all_course_rows(db_path: Path) -> List[Dict[str, Any]]:
    sql = (
        f"SELECT course_id, course_code, title, semester, ects, course_type, course_kind, "
        f"knowledge_area, weekly_hours, coordinator, course_url "
        f"FROM {DB_TABLE_COURSES}"
    )
    with _connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_matches_filters(row: Dict[str, Any], filters: Dict[str, str]) -> bool:
    for key, raw_value in filters.items():
        if key == "semester":
            if _semester_digits(row.get("semester")) != str(raw_value):
                return False
            continue

        if key == "course_code":
            if not _contains_norm(row.get("course_code"), raw_value):
                return False
            continue

        if key == "topic":
            if not (
                _contains_norm(row.get("title"), raw_value)
                or _contains_norm(row.get("knowledge_area"), raw_value)
            ):
                return False
            continue

        if key == "laboratory":
            if not _contains_norm(row.get("weekly_hours"), "(ε)"):
                return False
            continue

        if key in ("course_kind", "course_type", "coordinator", "title", "knowledge_area"):
            if not _contains_norm(row.get(key), raw_value):
                return False
            continue

    return True


def _filtered_course_rows(db_path: Path, filters: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = _fetch_all_course_rows(db_path)
    return [row for row in rows if _row_matches_filters(row, filters)]


def count_courses(db_path: Path, filters: Dict[str, str]) -> Dict[str, Any]:
    total = len(_filtered_course_rows(db_path, filters))
    return {"total_count": total}


def list_courses(db_path: Path, filters: Dict[str, str], limit: int = DEFAULT_LIST_LIMIT) -> Dict[str, Any]:
    rows = _filtered_course_rows(db_path, filters)
    rows.sort(
        key=lambda r: (
            int(_semester_digits(r.get("semester")) or 999),
            normalize_text(str(r.get("course_code") or "")),
            normalize_text(str(r.get("title") or "")),
        )
    )
    items = rows[: int(limit)]

    return {"rows": items, "row_count": len(items), "limit": int(limit)}


def sum_ects(db_path: Path, filters: Dict[str, str]) -> Dict[str, Any]:
    total = 0.0
    for row in _filtered_course_rows(db_path, filters):
        try:
            total += float(row.get("ects") or 0)
        except Exception:
            continue

    return {"total_ects": total}


def group_count_courses(db_path: Path, group_field: str, filters: Dict[str, str]) -> Dict[str, Any]:
    allowed_group_fields = {
        "semester": "semester",
        "course_type": "course_type",
        "course_kind": "course_kind",
        "knowledge_area": "knowledge_area",
        "coordinator": "coordinator",
    }
    field = allowed_group_fields.get(group_field, "semester")

    counts: Dict[str, int] = {}
    display_values: Dict[str, str] = {}
    for row in _filtered_course_rows(db_path, filters):
        raw = row.get(field)
        key = normalize_text(str(raw or ""))
        if not key:
            key = "-"
        counts[key] = counts.get(key, 0) + 1
        if key not in display_values:
            display_values[key] = str(raw or "-")

    groups = [
        {"group_value": display_values[k], "total_count": v}
        for k, v in counts.items()
    ]
    groups.sort(key=lambda g: (-int(g["total_count"]), normalize_text(str(g["group_value"]))))

    return {"group_by": field, "groups": groups, "row_count": len(groups)}


def get_distinct_values(db_path: Path, field: str, filters: Dict[str, str]) -> Dict[str, Any]:
    allowed_fields = {
        "semester": "semester",
        "course_type": "course_type",
        "course_kind": "course_kind",
        "knowledge_area": "knowledge_area",
        "coordinator": "coordinator",
    }
    db_field = allowed_fields.get(field)
    if not db_field:
        return {"field": field, "values": [], "row_count": 0}

    values_map: Dict[str, Any] = {}
    for row in _filtered_course_rows(db_path, filters):
        value = row.get(db_field)
        if value is None or str(value).strip() == "":
            continue
        key = normalize_text(str(value))
        if key not in values_map:
            values_map[key] = value

    values = list(values_map.values())
    values = sorted(values, key=lambda x: normalize_text(str(x)))
    return {"field": db_field, "values": values, "row_count": len(values)}


def _format_course_row(row: Dict[str, Any]) -> str:
    code = str(row.get("course_code") or "-").strip()
    title = str(row.get("title") or "Άγνωστος τίτλος").strip()
    semester = str(row.get("semester") or "-").strip()
    ects = row.get("ects")
    coordinator = str(row.get("coordinator") or "-").strip()
    return f"{code} | {title} | εξάμηνο {semester} | ECTS {ects} | συντονιστής {coordinator}"


def _build_natural_answer(operation: str, filters: Dict[str, str], payload: Dict[str, Any]) -> str:
    if operation == "course_entity_lookup":
        row = payload.get("row") or {}
        if row:
            return _build_course_entity_answer(row)
        return "Δεν βρέθηκε σχετική πληροφορία για το συγκεκριμένο μάθημα."

    if operation == "study_duration":
        total = int(payload.get("effective_semesters") or 0)
        years = float(payload.get("duration_years") or 0)
        years_text = str(int(years)) if years.is_integer() else f"{years:.1f}"
        if total > 0:
            return f"Η τυπική διάρκεια σπουδών είναι {total} εξάμηνα ({years_text} έτη)."
        return "Δεν υπάρχει διαθέσιμη πληροφορία για τη διάρκεια σπουδών."

    if operation == "count_courses":
        total = int(payload.get("total_count") or 0)
        return f"Βρέθηκαν συνολικά {total} μαθήματα με τα συγκεκριμένα φίλτρα."

    if operation == "sum_ects":
        total = float(payload.get("total_ects") or 0.0)
        if total.is_integer():
            total_text = str(int(total))
        else:
            total_text = f"{total:.2f}"
        return f"Το συνολικό άθροισμα ECTS είναι {total_text}."

    if operation == "group_count":
        groups = payload.get("groups") or []
        if not groups:
            return "Δεν βρέθηκαν ομαδοποιημένα αποτελέσματα για τα συγκεκριμένα φίλτρα."
        prefix = f"Κατανομή μαθημάτων ανά {payload.get('group_by')}:\n"
        lines = [f"- {g.get('group_value')}: {g.get('total_count')}" for g in groups]
        return prefix + "\n".join(lines)

    if operation == "distinct_values":
        values = payload.get("values") or []
        field = payload.get("field") or "field"
        if not values:
            return f"Δεν βρέθηκαν διαθέσιμες τιμές για το πεδίο {field}."
        text_values = ", ".join(str(v) for v in values)
        return f"Διακριτές τιμές για {field}: {text_values}."

    rows = payload.get("rows") or []
    if not rows:
        return "Δεν βρέθηκαν μαθήματα που να ταιριάζουν με τα φίλτρα."

    lines = ["Βρέθηκαν τα ακόλουθα μαθήματα:"]
    for row in rows:
        lines.append(f"- {_format_course_row(row)}")
    return "\n".join(lines)


def answer_course_query_structured(
    query: str,
    db_path: Path,
    *,
    min_confidence: float = 0.55,
    list_limit: int = DEFAULT_LIST_LIMIT,
) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None

    operation = _detect_operation(query)
    if operation is None:
        entity_match = find_best_course_entity_match(query, db_path)
        if entity_match is None or float(entity_match.get("score") or 0.0) < COURSE_ENTITY_MATCH_MIN_SCORE:
            return None
        operation = "course_entity_lookup"

    filters, topic_only_hint = _extract_filters(query)

    confidence = 0.55
    q_norm = normalize_text(query)
    if any(k in q_norm for k in ("how many", "count", "sum", "list", "show", "which", "ποσα", "πόσα", "ποια")):
        confidence += 0.2
    if filters:
        confidence += 0.15
    if operation in ("count_courses", "sum_ects", "group_count", "study_duration", "course_entity_lookup"):
        confidence += 0.1
    confidence = min(confidence, 0.98)

    if confidence < min_confidence:
        return {
            "answered": False,
            "should_fallback_to_rag": True,
            "reason": "low_confidence",
            "confidence": confidence,
            "operation": operation,
            "filters": filters,
            "query": query,
        }

    payload: Dict[str, Any]
    if operation == "count_courses":
        payload = count_courses(db_path, filters)
    elif operation == "course_entity_lookup":
        entity_match = find_best_course_entity_match(query, db_path)
        if entity_match is None:
            return {
                "answered": False,
                "should_fallback_to_rag": True,
                "reason": "course_not_found",
                "confidence": confidence,
                "operation": operation,
                "filters": filters,
                "query": query,
            }

        score = float(entity_match.get("score") or 0.0)
        if score < COURSE_ENTITY_MATCH_MIN_SCORE:
            return {
                "answered": False,
                "should_fallback_to_rag": True,
                "reason": "low_course_match_confidence",
                "confidence": score,
                "operation": operation,
                "filters": filters,
                "query": query,
            }

        payload = {
            "row": entity_match.get("row") or {},
            "match_score": score,
            "match_type": entity_match.get("match_type") or "fuzzy",
            "candidate": entity_match.get("candidate") or "",
        }
        confidence = max(confidence, score)
    elif operation == "study_duration":
        payload = _study_duration_from_courses(db_path)
    elif operation == "sum_ects":
        payload = sum_ects(db_path, filters)
    elif operation == "group_count":
        group_field = _detect_group_field(query)
        payload = group_count_courses(db_path, group_field, filters)
    elif operation == "distinct_values":
        payload = get_distinct_values(db_path, _detect_group_field(query), filters)
    else:
        payload = list_courses(db_path, filters, limit=list_limit)

    answer = _build_natural_answer(operation, filters, payload)

    should_fallback_to_rag = False
    if operation in ("list_courses", "group_count", "distinct_values"):
        row_count = int(payload.get("row_count") or 0)
        if row_count == 0 and topic_only_hint:
            # Topic-only misses are often better answered from unstructured docs.
            should_fallback_to_rag = True

    source_row = {}
    if payload.get("rows"):
        source_row = payload.get("rows")[0] or {}
    elif payload.get("row"):
        source_row = payload.get("row") or {}

    return {
        "answered": True,
        "query": query,
        "confidence": confidence,
        "operation": operation,
        "filters": filters,
        "source_table": DB_TABLE_COURSES,
        "row": source_row,
        "payload": payload,
        "answer": answer,
        "should_fallback_to_rag": should_fallback_to_rag,
    }
