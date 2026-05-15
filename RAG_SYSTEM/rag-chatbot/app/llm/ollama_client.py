import requests
from typing import Optional

class OllamaClient:
    """Tiny wrapper for a local Ollama-like service (placeholder)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, model: str = "llama2") -> str:
        url = f"{self.base_url}/api/generate"
        try:
            resp = requests.post(url, json={"model": model, "prompt": prompt}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "")
        except Exception:
            return ""
