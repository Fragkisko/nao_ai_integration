import json
import spacy

import fitz
from pathlib import Path
import argparse
import sys
import re
import unicodedata


import re


def normalize_section_header(header: str) -> str:
    return re.sub(r"\s+", " ", header.strip())


def is_section_title(title: str, allow_top_level: bool = False) -> bool:
    title = title.strip()
    if not title or len(title.split()) > 12:
        return False
    if any(char in title for char in "():;"):
        return False
    if allow_top_level and any(char.isdigit() for char in title):
        return False
    return True


def extract_section_prefix(text: str):
    stripped = text.lstrip()
    if not stripped:
        return None, "", 0

    multiline_match = re.match(
        r"^(?P<number>(?:\d+(?:\.\d+)+|\d+\.))\s*\n\s*(?P<title>[^\n]{1,120})(?P<rest>(?:\n.*)?)$",
        stripped,
    )
    if multiline_match:
        number = multiline_match.group("number")
        title = multiline_match.group("title").strip()
        if is_section_title(title, allow_top_level=number.endswith(".")):
            header = normalize_section_header(f"{number} {title}")
            consumed = multiline_match.start("rest") if multiline_match.group("rest") else len(stripped)
            remainder = stripped[consumed:].lstrip()
            return header, remainder, len(text) - len(stripped) + consumed + (len(stripped[consumed:]) - len(remainder))

    inline_match = re.match(
        r"^(?P<number>(?:\d+(?:\.\d+)+|\d+\.))\s+(?P<title>[^\n]{1,120})(?P<rest>(?:\n.*)?)$",
        stripped,
    )
    if inline_match:
        number = inline_match.group("number")
        title = inline_match.group("title").strip()
        if is_section_title(title, allow_top_level=number.endswith(".")):
            header = normalize_section_header(f"{number} {title}")
            consumed = inline_match.start("rest") if inline_match.group("rest") else len(stripped)
            remainder = stripped[consumed:].lstrip()
            return header, remainder, len(text) - len(stripped) + consumed + (len(stripped[consumed:]) - len(remainder))

    return None, text, 0


def is_bullet_sentence(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    if stripped.startswith(("να ", "•", "-", "–")):
        return True
    return bool(re.match(r"^(\(?[α-ωa-z0-9]+[\).])\s+", stripped))

def normalize_text(text: str) -> str:
    # lowercase (ΟΚ να το κρατήσεις, αλλά optional)
    text = text.lower()

    # fix hyphenation από PDFs (ΠΟΛΥ σημαντικό)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    lines = text.splitlines()
    norm_lines = []

    bullet_patterns = [
        r"^[-•‣▪●◦‒–—―·∙*]+$",
        r"^\d+\.$",
        r"^\d+\)$",
        r"^\([a-zA-Z]\)$",
        r"^\([ivxlcdm]+\)$",
        r"^[a-zA-Z]\.$",
        r"^[ivxlcdm]+\.$",
    ]

    for line in lines:
        line = line.strip()

        # skip empty
        if not line:
            continue

        # page numbers
        if re.fullmatch(r"\d+", line):
            continue

        # σελ / page
        if re.match(r"^(σελ|page)\s*\d+", line):
            continue

        # separators
        if re.match(r"^\s*-+\s*$", line):
            continue

        # bullets μόνοι τους
        if any(re.fullmatch(pat, line) for pat in bullet_patterns):
            continue

        norm_lines.append(line)

    text = "\n".join(norm_lines)

    text = re.sub(r"[ \t]+", " ", text)

    # normalize newlines 
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()

def split_sentences_safe(text, nlp, batch_size=5000):
    sentences = []

    for i in range(0, len(text), batch_size):
        part = text[i:i+batch_size]
        doc = nlp(part)
        for sent in doc.sents:
            raw_text = sent.text
            stripped_text = raw_text.strip()
            if not stripped_text:
                continue
            leading_ws = len(raw_text) - len(raw_text.lstrip())
            trailing_ws = len(raw_text) - len(raw_text.rstrip())
            start = i + sent.start_char + leading_ws
            end = i + sent.end_char - trailing_ws
            sentences.append({
                "text": stripped_text,
                "start": start,
                "end": end,
            })

    return sentences


    

# --- New helper for cleaning chunk text ---
def clean_chunk_text(text):
    text = text.strip()
    text = re.sub(r"^[\.,\-]\s*", "", text)  # remove weird starts
    text = re.sub(r"[\.,\-]\s*$", "", text)  # remove weird ends
    return text


def is_valid_chunk(text: str) -> bool:
    text = text.strip()

    # πολύ μικρό
    if len(text) < 30:
        return False

    # μόνο symbols
    if re.fullmatch(r"[\W_]+", text):
        return False

    # πολύ λίγα γράμματα
    letters = sum(c.isalpha() for c in text)
    if letters < 10:
        return False

    return True

# --- New helper for extracting section headers ---
def extract_section(line):
    header, _, _ = extract_section_prefix(line)
    return header

def chunk_sentences(sentences, max_chars=300, max_words=None, overlap_sents=1, verbose=False):
    """
    Χωρίζει λίστα προτάσεων σε chunks με όριο χαρακτήρων, χωρίς να κόβει προτάσεις, με overlap σε προτάσεις.
    overlap_sents: π.χ. 1 για 1 πρόταση overlap (αντί για overlap σε λέξεις)
    Επιστρέφει λίστα από strings (chunks)
    Αν verbose=True, εκτυπώνει πληροφορίες για τα chunks
    """
    chunks = []
    i = 0
    n = len(sentences)
    while i < n:
        chunk_sents = []
        char_count = 0
        word_count = 0
        j = i
        while j < n:
            sent_text = sentences[j]
            sent_words = len(sent_text.split())
            sent_chars = len(sent_text)
            exceeds_chars = chunk_sents and char_count + 1 + sent_chars > max_chars
            exceeds_words = max_words is not None and chunk_sents and word_count + sent_words > max_words
            if exceeds_chars or exceeds_words:
                break
            chunk_sents.append(sent_text)
            char_count += sent_chars + (1 if len(chunk_sents) > 1 else 0)
            word_count += sent_words
            j += 1
        if chunk_sents:
            chunk_text = clean_chunk_text(" ".join(chunk_sents))
            chunks.append(chunk_text)
            if verbose:
                print(f"Chunk {len(chunks)}: {char_count} chars, {word_count} words, sentences {i+1}-{j}")
        if j == n:
            break
        # overlap: go back by overlap_sents (in sentences)
        i = max(j - overlap_sents, i+1)  # avoid infinite loop
    return chunks

def chunk_text_with_spacy(text, nlp, max_chars=300, max_words=None, overlap_sents=1, verbose=False):
    """
    Κάνει sentence split με spaCy και μετά chunking με overlap σε προτάσεις.
    Τα section headers μπαίνουν μόνο στο metadata και όχι στο chunk text.
    Επιστρέφει λίστα chunks.
    """
    raw_sentences = split_sentences_safe(text, nlp)
    content_units = []
    current_section = None
    for sentence in raw_sentences:
        section_header, remainder, consumed_chars = extract_section_prefix(sentence["text"])
        if section_header:
            current_section = section_header
        if not remainder.strip():
            continue

        start = sentence["start"] + consumed_chars
        while start < sentence["end"] and text[start].isspace():
            start += 1

        content_units.append({
            "text": remainder.strip(),
            "section": current_section,
            "start": start,
            "end": sentence["end"],
            "is_bullet": is_bullet_sentence(remainder),
        })

    chunks = []
    i = 0
    n = len(content_units)
    while i < n:
        chunk_units = []
        chunk_texts = []
        char_count = 0
        word_count = 0
        j = i
        chunk_section = None
        while j < n:
            unit = content_units[j]
            unit_text = unit["text"]
            unit_words = len(unit_text.split())
            unit_chars = len(unit_text)
            if chunk_units and unit["is_bullet"]:
                break

            exceeds_chars = chunk_units and char_count + 1 + unit_chars > max_chars
            exceeds_words = max_words is not None and chunk_units and word_count + unit_words > max_words
            if exceeds_chars or exceeds_words:
                break

            chunk_units.append(unit)
            chunk_texts.append(unit_text)
            if not chunk_section:
                chunk_section = unit["section"]
            char_count += unit_chars + (1 if len(chunk_texts) > 1 else 0)
            word_count += unit_words
            j += 1

            if unit["is_bullet"]:
                break

        if chunk_units:
            chunk_text = clean_chunk_text(" ".join(chunk_texts))
            # skip garbage / too-small chunks
            if not is_valid_chunk(chunk_text):
                if verbose:
                    print(f"Skipping invalid/garbage chunk #{len(chunks)+1}: '{chunk_text[:40]}...'")
            else:
                chunk_start = chunk_units[0]["start"]
                chunk_end = chunk_units[-1]["end"]
                chunks.append({
                    "text": chunk_text,
                    "section": chunk_section,
                    "start": chunk_start,
                    "end": chunk_end,
                    "retrieval_text": f"{chunk_section}\n{chunk_text}" if chunk_section else chunk_text,
                })
            if verbose:
                print(f"Chunk {len(chunks)}: {char_count} chars, {word_count} words, sentences {i+1}-{j}, section: {chunk_section}")
        if j == n:
            break
        i = max(j - overlap_sents, i+1)
    return chunks
    
def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    raw_text = "\n".join(parts)
    return normalize_text(raw_text)

def process_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    for pdf in pdf_files:
        try:
            text = extract_pdf_text(pdf)
        except Exception as e:
            print(f"Error extracting {pdf}: {e}", file=sys.stderr)
            continue

        rel = pdf.relative_to(input_dir)
        out_subdir = output_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / (pdf.stem + ".txt")

        if out_file.exists() and not overwrite:
            print(f"Skipping existing {out_file}")
            continue

        out_file.write_text(text, encoding="utf-8")
        print(f"Wrote {out_file}")

def main():

    parser = argparse.ArgumentParser(description="Extract text from PDFs or chunk .txt files.")
    parser.add_argument("input_dir", help="Input directory containing PDF or TXT files")
    parser.add_argument("-o", "--output", help="Output directory for extracted or chunked files (default: extracted_text or extracted_chunks inside input_dir)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--chunk-texts", action="store_true", help="Chunk all .txt files in input_dir instead of extracting from PDFs")
    parser.add_argument("--max-chars", type=int, default=300, help="Max characters per chunk (default: 300)")
    parser.add_argument("--max-words", type=int, default=None, help="Optional max words per chunk (legacy compatibility)")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap ratio (default: 0.1 = 10%)")
    parser.add_argument("--spacy-model", type=str, default="el_core_news_sm", help="spaCy model for sentence splitting (default: el_core_news_sm)")
    parser.add_argument("--verbose", action="store_true", help="Print chunking info")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = input_dir / ("extracted_chunks" if args.chunk_texts else "extracted_text")

    if args.chunk_texts:
        nlp = spacy.load(args.spacy_model)
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return
        chunk_preview = []
        global_chunk_id = 1
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8")
            # Use new chunking with section detection and cleaned text
            chunks = chunk_text_with_spacy(
                text, nlp,
                max_chars=args.max_chars,
                max_words=args.max_words,
                overlap_sents=1,  # <<<< set overlap to 1 sentence
                verbose=args.verbose
            )
            for i, chunk_data in enumerate(chunks):
                chunk = chunk_data["text"]
                section_title = chunk_data["section"]
                chunk_id = global_chunk_id
                out_file = output_dir / f"{txt_file.stem}_chunk_{i+1}.txt"
                out_file.write_text(chunk, encoding="utf-8")
                chunk_preview.append({
                    "id": chunk_id,
                    "text": chunk,
                    "retrieval_text": chunk_data["retrieval_text"],
                    "source": txt_file.name,
                    "chunk_index": i + 1,
                    "start_char": chunk_data["start"],
                    "end_char": chunk_data["end"],
                    "section": section_title,
                    "language": "el"
                })
                global_chunk_id += 1
            print(f"Chunked {txt_file.name} into {len(chunks)} chunks. Saved to {output_dir}")
        # Write chunk preview file
        preview_path = output_dir / "chunk_preview.json"
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(chunk_preview, f, ensure_ascii=False, indent=2)
        print(f"Chunk preview file written to {preview_path}")
    else:
        process_directory(input_dir, output_dir, overwrite=args.force)

if __name__ == "__main__":
    main()