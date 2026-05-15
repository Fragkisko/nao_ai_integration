import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
