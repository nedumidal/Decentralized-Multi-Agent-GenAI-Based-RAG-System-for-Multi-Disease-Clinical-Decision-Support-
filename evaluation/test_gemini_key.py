"""
Quick Gemini key test (takes ~2 seconds, does not load BioBERT or any agents).

Run:  python evaluation/test_gemini_key.py              (tests GOOGLE_API_KEY from .env)
      python evaluation/test_gemini_key.py MY_OTHER_VAR (tests a different variable name)

Prints only the HTTP status and Google's error reason. The key itself is never printed.
"""
import os
import sys
import requests
from dotenv import dotenv_values

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# utf-8-sig so a Windows byte-order mark at the start of .env cannot break the first variable name
env = dotenv_values(os.path.join(ROOT, ".env"), encoding="utf-8-sig")

name = sys.argv[1] if len(sys.argv) > 1 else "GOOGLE_API_KEY"
key = (env.get(name) or "").strip()
if not key:
    sys.exit(f"{name} not found in .env (variables present: {', '.join(env.keys())})")

MODEL = "gemini-3.6-flash"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
resp = requests.post(
    url,
    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    json={"contents": [{"parts": [{"text": "Reply with the single letter B."}]}]},
    timeout=30,
)

print(f"Variable tested : {name}  (length {len(key)}, starts with '{key[:3]}')")
print(f"Model           : {MODEL}")
print(f"HTTP status     : {resp.status_code}")

if resp.status_code == 200:
    try:
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = resp.text[:200]
    print(f"Reply           : {text.strip()[:80]}")
    print("\nRESULT: key works. You can rerun evaluation/model_comparison.py")
else:
    body = resp.text.replace(key, "<key>")
    try:
        err = resp.json().get("error", {})
        reason = next((d.get("reason") for d in err.get("details", []) if d.get("reason")), None)
        print(f"Google status   : {err.get('status')}")
        print(f"Reason          : {reason}")
        print(f"Message         : {str(err.get('message'))[:300].replace(key, '<key>')}")
    except Exception:
        print(body[:400])
    print("\nRESULT: key does NOT work yet (see reason above).")
