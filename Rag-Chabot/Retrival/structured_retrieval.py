"""Structured retrieval helpers for staff-related queries from SQLite DB.

This module is intentionally independent from embedding-based retrieval.
It answers professor/staff questions directly from `iee_ihu_staff`.
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


DB_TABLE_STAFF = "iee_ihu_staff"
DB_TABLE_COURSES = "iee_ihu_courses"

# Very small stopword list for extracting likely name tokens from Greek queries.
_QUERY_STOPWORDS = {
	"ποιος",
	"ποια",
	"ποιο",
	"ειναι",
	"είναι",
	"ο",
	"η",
	"το",
	"του",
	"της",
	"των",
	"στο",
	"στη",
	"στην",
	"για",
	"με",
	"και",
	"τι",
	"που",
	"ποιες",
	"ποιους",
	"καθηγητης",
	"καθηγητή",
	"καθηγητής",
	"διδασκει",
	"διδάσκει",
	"γραφειο",
	"γραφείο",
	"μαθημα",
	"μάθημα",
	"κωδικος",
	"κωδικός",
	"συντονιζει",
	"συντονίζει",
	"συντονιστης",
	"συντονιστής",
}


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
	"""Decode SQLite text robustly for mixed encodings (utf-8/cp1253)."""
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


def should_route_to_staff_db(query: str) -> bool:
	"""Heuristic router for staff/professor style questions."""
	q = normalize_text(query)
	if not q:
		return False
	keywords = (
		"ποιος",
		"ποια",
		"καθηγητ",
		"διδασκ",
		"γραφει",
		"γραφείο",
		"office",
		"staff",
	)
	return any(k in q for k in keywords)


def should_route_to_courses_db(query: str) -> bool:
	"""Heuristic router for course-related questions."""
	q = normalize_text(query)
	if not q:
		return False
	keywords = (
		"μαθημα",
		"μάθημα",
		"course",
		"κωδικ",
		"ects",
		"συντονιζ",
		"coordinator",
		"περιεχομενο",
		"περιεχόμενο",
	)
	return any(k in q for k in keywords) or bool(re.search(r"\b\d{4}\b", q))


def _query_tokens(query: str) -> List[str]:
	q = normalize_text(query)
	tokens = re.findall(r"[\wά-ώα-ωΑ-Ω]+", q, flags=re.UNICODE)
	return [t for t in tokens if len(t) >= 3 and t not in _QUERY_STOPWORDS]


def _fetch_staff_rows(db_path: Path) -> List[Dict[str, Any]]:
	if not db_path.exists():
		return []
	conn = sqlite3.connect(str(db_path))
	conn.text_factory = bytes
	conn.row_factory = sqlite3.Row
	try:
		cur = conn.cursor()
		cur.execute(f"SELECT * FROM {DB_TABLE_STAFF}")
		return [_row_to_dict(r) for r in cur.fetchall()]
	finally:
		conn.close()


def _fetch_course_rows(db_path: Path) -> List[Dict[str, Any]]:
	if not db_path.exists():
		return []
	conn = sqlite3.connect(str(db_path))
	conn.text_factory = bytes
	conn.row_factory = sqlite3.Row
	try:
		cur = conn.cursor()
		cur.execute(f"SELECT * FROM {DB_TABLE_COURSES}")
		return [_row_to_dict(r) for r in cur.fetchall()]
	finally:
		conn.close()


def _score_staff_row(row: Dict[str, Any], tokens: List[str]) -> float:
	full_name = normalize_text(row.get("full_name") or "")
	first_name = normalize_text(row.get("first_name") or "")
	last_name = normalize_text(row.get("last_name") or "")
	courses = normalize_text(row.get("courses") or "")
	office = normalize_text(row.get("office_location") or "")

	score = 0.0
	for tok in tokens:
		if tok in full_name:
			score += 3.0
		if tok in first_name or tok in last_name:
			score += 2.0
		if tok in courses:
			score += 0.8
		if tok in office:
			score += 0.5

	# Bonus for matching both first and last name signals.
	if tokens and any(t in first_name for t in tokens) and any(t in last_name for t in tokens):
		score += 2.0
	return score


def find_best_staff_match(query: str, db_path: Path) -> Optional[Tuple[Dict[str, Any], float]]:
	rows = _fetch_staff_rows(db_path)
	if not rows:
		return None

	tokens = _query_tokens(query)
	if not tokens:
		return None

	ranked = [(r, _score_staff_row(r, tokens)) for r in rows]
	ranked.sort(key=lambda x: x[1], reverse=True)
	best_row, best_score = ranked[0]
	if best_score <= 0:
		return None
	return best_row, best_score


def build_staff_answer(staff_row: Dict[str, Any]) -> str:
	full_name = (staff_row.get("full_name") or "Το μέλος ΔΕΠ").strip()
	academic_rank = (staff_row.get("academic_rank") or "μέλος ΔΕΠ").strip()
	roles = (staff_row.get("administrative_roles") or "").strip()
	courses = (staff_row.get("courses") or "").strip()
	office = (staff_row.get("office_location") or "").strip()

	parts = [f"Ο/Η {full_name} είναι {academic_rank} του τμήματος."]
	if roles:
		parts.append(f"Διοικητικός ρόλος: {roles}.")
	if courses:
		parts.append(f"Διδάσκει τα μαθήματα: {courses}.")
	if office:
		parts.append(f"Το γραφείο βρίσκεται στο: {office}.")
	return " ".join(parts)


def answer_staff_query(query: str, db_path: Path, min_score: float = 3.0) -> Optional[Dict[str, Any]]:
	"""Return structured answer dict for staff query or None if no reliable hit."""
	match = find_best_staff_match(query, db_path)
	if match is None:
		return None

	row, score = match
	if score < min_score:
		return None

	answer = build_staff_answer(row)
	return {
		"query": query,
		"score": score,
		"source_table": DB_TABLE_STAFF,
		"staff_id": row.get("staff_id"),
		"row": row,
		"answer": answer,
	}


def _extract_course_codes(query: str) -> List[str]:
	return re.findall(r"\b(\d{4})\b", query or "")


def _score_course_row(row: Dict[str, Any], tokens: List[str], codes: List[str]) -> float:
	course_code = str(row.get("course_code") or "")
	title = normalize_text(row.get("title") or "")
	coordinator = normalize_text(row.get("coordinator") or "")
	content = normalize_text(row.get("content") or "")
	knowledge_area = normalize_text(row.get("knowledge_area") or "")

	score = 0.0
	for code in codes:
		if course_code == code:
			score += 8.0
		elif course_code.startswith(code):
			score += 5.0

	for tok in tokens:
		if tok in title:
			score += 2.0
		if tok in coordinator:
			score += 1.5
		if tok in knowledge_area:
			score += 0.8
		if tok in content:
			score += 0.5

	return score


def find_best_course_match(query: str, db_path: Path) -> Optional[Tuple[Dict[str, Any], float]]:
	rows = _fetch_course_rows(db_path)
	if not rows:
		return None

	tokens = _query_tokens(query)
	codes = _extract_course_codes(query)
	if not tokens and not codes:
		return None

	ranked = [(r, _score_course_row(r, tokens, codes)) for r in rows]
	ranked.sort(key=lambda x: x[1], reverse=True)
	best_row, best_score = ranked[0]
	if best_score <= 0:
		return None
	return best_row, best_score


def _course_query_intent(query: str) -> str:
	q = normalize_text(query)
	if any(k in q for k in ("συντονιζ", "coordinator", "υπευθυνος", "υπεύθυνος")):
		return "coordinator"
	if any(k in q for k in ("ects", "μοναδες", "μονάδες")):
		return "ects"
	if any(k in q for k in ("ωρες", "ώρες", "weekly")):
		return "hours"
	return "overview"


def build_course_answer(course_row: Dict[str, Any], intent: str = "overview") -> str:
	code = str(course_row.get("course_code") or "-").strip()
	title = (course_row.get("title") or "Άγνωστος τίτλος").strip()
	semester = (course_row.get("semester") or "-").strip()
	ects = course_row.get("ects")
	course_type = (course_row.get("course_type") or "").strip()
	course_kind = (course_row.get("course_kind") or "").strip()
	knowledge_area = (course_row.get("knowledge_area") or "").strip()
	weekly_hours = (course_row.get("weekly_hours") or "").strip()
	coordinator = (course_row.get("coordinator") or "").strip()
	content = (course_row.get("content") or "").strip()
	course_url = (course_row.get("course_url") or "").strip()

	if intent == "coordinator":
		if coordinator:
			return f"Ο συντονιστής του μαθήματος {code} ({title}) είναι ο/η {coordinator}."
		return f"Δεν υπάρχει διαθέσιμη πληροφορία συντονιστή για το μάθημα {code} ({title})."

	if intent == "ects":
		if ects is not None:
			return f"Το μάθημα {code} ({title}) έχει {ects} ECTS."
		return f"Δεν υπάρχει διαθέσιμη πληροφορία ECTS για το μάθημα {code} ({title})."

	if intent == "hours":
		if weekly_hours:
			return f"Το μάθημα {code} ({title}) έχει εβδομαδιαίες ώρες: {weekly_hours}."
		return f"Δεν υπάρχει διαθέσιμη πληροφορία εβδομαδιαίων ωρών για το μάθημα {code} ({title})."

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


def answer_course_query(query: str, db_path: Path, min_score: float = 2.5) -> Optional[Dict[str, Any]]:
	"""Return structured answer dict for course query or None if no reliable hit."""
	match = find_best_course_match(query, db_path)
	if match is None:
		return None

	row, score = match
	if score < min_score:
		return None

	intent = _course_query_intent(query)
	answer = build_course_answer(row, intent=intent)
	return {
		"query": query,
		"score": score,
		"source_table": DB_TABLE_COURSES,
		"course_id": row.get("course_id"),
		"row": row,
		"intent": intent,
		"answer": answer,
	}

