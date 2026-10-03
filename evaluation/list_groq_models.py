"""
List the Groq model IDs your API key can actually use right now.

Why: Groq retires models (Llama-3.3-70B and Llama-3.1-8B were shut down for free/developer
accounts on 2026-08-16). Run this before adding a model to evaluation/model_comparison.py.

Run:  python evaluation/list_groq_models.py
Your key is read from .env and is never printed.
"""
import os
import sys
import requests
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))

key = os.getenv("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY not found in .env")

resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=20,
)
if resp.status_code != 200:
    sys.exit(f"Groq returned HTTP {resp.status_code}: {resp.text[:300]}")

models = resp.json().get("data", [])
skip = ("whisper", "tts", "guard", "playai", "orpheus", "distil-whisper")
chat = sorted(m for m in models if not any(s in m["id"].lower() for s in skip))
print(f"{len(chat)} chat models available to your key:\n")
for m in chat:
    ctx = m.get("context_window", "?")
    print(f"  {m['id']:<55} context={ctx}  owner={m.get('owned_by', '?')}")
