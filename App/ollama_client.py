import requests

def ollama_chat(model: str, messages, temperature: float = 0.7):
    """
    messages: list[{"role": "system"|"user"|"assistant", "content": str}]
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]