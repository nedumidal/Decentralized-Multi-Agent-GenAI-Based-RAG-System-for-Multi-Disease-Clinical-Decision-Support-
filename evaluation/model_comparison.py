"""
Multi-Model Comparison for MARC-Clinical

Purpose: produce the per-agent x per-model accuracy/efficiency table the mentor
requested. Uses a 20-question sample - the same size the base paper itself used
for its own ablation study (Table 7).

Models compared:
  - Gemini 3.6 Flash    (Google)
  - Groq gpt-oss-120b   (OpenAI open-weight, via Groq, free tier)
  - Groq gpt-oss-20b    (OpenAI open-weight, via Groq, free tier, smaller/faster)
  - Groq models 4 and 5  (auto-selected at startup from the models your Groq key can
                          actually use; the list is printed when the script starts)

For each model, each of the 4 specialist agents independently analyzes the SAME 20
MedQA questions. Records per-agent accuracy, per-agent average response time, and a
confidence-weighted vote across the 4 agents as a cheap "ensemble" proxy (no separate
LLM fusion call; the full MARC pipeline is what medqa_benchmark.py measures).

Quota-safe behaviour (important):
  - If ANY agent call fails (daily token limit, empty billing balance, network), that
    question is NOT recorded. The model is paused cleanly and the script moves on to the
    next model. A failed call is never scored as a wrong answer.
  - Progress is saved after every completed question and resumes mid-model on rerun.
    Groq's free tier allows ~200k tokens/day per model, so each Groq model needs
    roughly 2 days: just rerun the same command each day until all models are complete.
"""
import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
# override=True: the .env file wins over any GOOGLE_API_KEY / GROQ_API_KEY already set in the
# Windows environment. (Without it, an old system-level key silently beat the .env key.)
# utf-8-sig: Windows editors often save .env with a hidden byte-order mark, which otherwise glues
# itself onto the first variable name (GOOGLE_API_KEY) so it never loads.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True, encoding="utf-8-sig")
_gk = os.getenv("GOOGLE_API_KEY") or ""
print(f"[keys] GOOGLE_API_KEY in use: starts with '{_gk[:3]}', length {len(_gk)} (key itself never printed)")

# Gemini key: the agents read GOOGLE_API_KEY (your original, now-funded key). The second key
# (GOOGLE_API_KEY2) is ignored unless you explicitly set MARC_USE_KEY2=1.
if os.getenv("MARC_USE_KEY2") == "1" and os.getenv("GOOGLE_API_KEY2"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY2"]
    print("Using GOOGLE_API_KEY2 for Gemini calls.")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

from agents.cardiology_agent import CardiologyAgent
from agents.diabetology_agent import DiabetologyAgent
from agents.nephrology_agent import NephrologyAgent
from agents.pharmacology_agent import PharmacologyAgent

KB = {
    "cardiology":  os.path.join(ROOT, "knowledge_bases/cardiology/aha_heart_failure.pdf"),
    "diabetology": os.path.join(ROOT, "knowledge_bases/diabetology/ada_standards_2025.pdf"),
    "nephrology":  os.path.join(ROOT, "knowledge_bases/nephrology/kdigo_ckd_2024.pdf"),
    "pharmacology":os.path.join(ROOT, "knowledge_bases/pharmacology/who_essential_medicines.pdf"),
}

NUM_QUESTIONS = 20  # matches base paper's own Table 7 ablation sample size
SPECIALTIES = ["cardiology", "nephrology", "diabetology", "pharmacology"]

BASE_MODELS = [
    # "pace" = seconds between agent calls (default 2). Small-TPM free models need slower pacing.
    {"label": "Gemini 3.6 Flash",  "provider": "gemini", "groq_model": None},
    {"label": "Groq gpt-oss-120b", "provider": "groq",   "groq_model": "openai/gpt-oss-120b"},
    {"label": "Groq gpt-oss-20b",  "provider": "groq",   "groq_model": "openai/gpt-oss-20b"},
]

# Set MARC_SKIP_GEMINI=1 (PowerShell:  $env:MARC_SKIP_GEMINI=1 ) to leave Gemini out, e.g.
# while its key/billing is blocked. Its saved progress is kept.
if os.getenv("MARC_SKIP_GEMINI") == "1":
    BASE_MODELS = [m for m in BASE_MODELS if m["provider"] != "gemini"]
    print("MARC_SKIP_GEMINI=1 -> Gemini left out of this run.")

TOTAL_TARGET_MODELS = 5

# Preferred extra Groq models, in order. Only ones your key can actually use are picked.
# (Groq retires models often: Llama-3.3-70B / Llama-3.1-8B were shut down 2026-08-16.)
CANDIDATES = [
    ("Groq Qwen3.8-27B",     "qwen/qwen3.8-27b", 30),
    ("Groq Llama-4-Scout",    "meta-llama/llama-4-scout-17b-16e-instruct", 5),
    ("Groq Kimi-K2",          "moonshotai/kimi-k2-instruct-0905", 5),
    ("Groq Qwen3-32B",        "qwen/qwen3-32b", 10),
    ("Groq Qwen3.6-27B",      "qwen/qwen3.6-27b", 5),
    ("Groq Llama-4-Maverick", "meta-llama/llama-4-maverick-17b-128e-instruct", 5),
]
_NOT_CHAT = ("whisper", "tts", "guard", "playai", "orpheus", "safeguard", "compound", "embed",
             "allam")  # allam-2-7b is an Arabic-focused 7B model: unusable for English MedQA

# Per-model Groq request settings. Qwen3.8-27B's free tier enforces 1000 output tokens/minute
# and rejects any request whose max output could exceed it (HTTP 429 "Request too large").
# Its default reasoning mode is already "none", so capping the reply length is enough.
EXTRA_GROQ_KWARGS = {"qwen/qwen3.8-27b": {"max_tokens": 700}}


def _available_groq_models():
    """Ask Groq which model IDs this key can use. Returns a sorted list, or None on failure."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("[models] GROQ_API_KEY not found in .env - cannot pick extra Groq models.")
        return None
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}", "User-Agent": "marc-clinical/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        return sorted(m["id"] for m in data.get("data", []))
    except Exception as e:
        print(f"[models] Could not list Groq models ({type(e).__name__}); extra models skipped.")
        return None


def _pick_extra_models(n_needed):
    if n_needed <= 0:
        return []
    avail = _available_groq_models()
    if avail is None:
        return []
    print("\nGroq models available to your key:")
    for mid in avail:
        print(f"   {mid}")
    used = {m["groq_model"] for m in BASE_MODELS}
    picked = []
    for label, mid, pace in CANDIDATES:
        if len(picked) >= n_needed:
            break
        if mid in avail and mid not in used:
            picked.append({"label": label, "provider": "groq", "groq_model": mid, "pace": pace,
                           "groq_kwargs": EXTRA_GROQ_KWARGS.get(mid, {})})
    if len(picked) < n_needed:  # fill from anything else that is a normal chat model
        chosen = used | {m["groq_model"] for m in picked}
        for mid in avail:
            if len(picked) >= n_needed:
                break
            if mid in chosen or any(s in mid.lower() for s in _NOT_CHAT):
                continue
            picked.append({"label": ("Groq " + mid.split("/")[-1])[:19], "provider": "groq",
                           "groq_model": mid, "pace": 5})
    print("\nExtra models selected for this comparison:")
    for m in picked:
        print(f"   {m['label']}  ->  {m['groq_model']}")
    if len(picked) < n_needed:
        print(f"   (only {len(picked)} of {n_needed} extra models available)")
    print()
    return picked


MODELS = BASE_MODELS + _pick_extra_models(TOTAL_TARGET_MODELS - len(BASE_MODELS))


PROGRESS_PATH = "evaluation/model_comparison_progress.json"


def format_question(q):
    options = q["options"]
    opt_str = "\n".join([f"  {k}) {v}" for k, v in options.items()])
    return (f"Medical Question (USMLE-style):\n{q['question']}\n\nOptions:\n{opt_str}\n\n"
            "Select the best answer. Start your response with the answer letter (A/B/C/D) only.")


def get_correct_answer(q):
    for field in ["answer_idx", "answer"]:
        if field in q and q[field]:
            val = str(q[field]).strip().upper()
            if val in ["A", "B", "C", "D"]:
                return val
    raise ValueError(f"No valid answer key found in question record: {list(q.keys())}")


def extract_answer(text):
    """Robust letter extraction. Returns 'X' when no clear answer letter is found.
    Never guesses from stray letters inside words (old fallback scanned the first 80
    characters and could silently score 'A' from words like 'ANSWER')."""
    if not text:
        return "X"
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # drop reasoning blocks (Qwen/Kimi-style models)
    if not t or t.lower().startswith("unavailable"):  # agent's failure marker only, not the word inside a real analysis
        return "X"
    # 1. Answer on the first non-empty line: "B", "B)", "**B**", "Answer: B"
    first = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"^[\s\*\#\(\[]*(?:(?i:(?:best |correct |final )?answer)\s*(?i:is)?\s*[:\-]?\s*)?[\*\(\[]*([ABCD])(?=\s*(?:[\)\.\:\*\]]|-|$))", first)
    if m:
        return m.group(1)
    # 2. Explicit marker anywhere: "Answer: B", "the answer is (B)"
    m = re.search(r"(?i:(?:best |correct |final )?answer\s*(?:is)?)\s*[:\-]?\s*[\*\(\[]*\s*([ABCD])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    # 3. A single distinct parenthesised letter, e.g. "(D)"
    letters = set(re.findall(r"\(([ABCD])\)", t))
    if len(letters) == 1:
        return letters.pop()
    return "X"


def _redact(s):
    """Hide API keys that Google/Groq echo back inside error messages."""
    s = re.sub(r"api_key:[^\s'\"\\]+", "api_key:<redacted>", s)
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "<redacted>", s)
    s = re.sub(r"AQ\.[0-9A-Za-z_\-]{20,}", "<redacted>", s)
    s = re.sub(r"gsk_[0-9A-Za-z]{20,}", "<redacted>", s)
    return s


def analyze_with_retry(agent, case, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return agent.analyze(case)
        except Exception as e:
            last_error = e
            es = str(e)
            print(f"    [{agent.name}] error: {_redact(es)[:600]}")
            if any(t in es for t in ("PERMISSION_DENIED", "403", "402", "model_not_found", "CONSUMER_SUSPENDED")):
                break  # retrying cannot fix these
            wait = 20 if ("503" in es or "UNAVAILABLE" in es) else (40 if "429" in es else 10)
            if attempt == max_retries - 1:
                break
            print(f"    Retry {attempt+1} in {wait}s...")
            time.sleep(wait)
    print(f"    [{agent.name}] giving up. Last error: {_redact(str(last_error))[:600]}")
    return {"agent": agent.name, "specialty": agent.specialty,
            "analysis": "Unavailable.", "confidence": 0.0}


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(results):
    os.makedirs("evaluation", exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def questions_done(results, label):
    return results.get(label, {}).get("questions_done", 0)


def new_model_record():
    rec = {sp: {"correct": 0, "total": 0, "total_time": 0.0} for sp in SPECIALTIES}
    rec["ensemble_vote"] = {"correct": 0, "total": 0}
    rec["questions_done"] = 0
    return rec


def run_one_model(model_cfg, questions, results):
    """Returns True if this model finished all questions, False if paused (quota etc.)."""
    label = model_cfg["label"]
    done = questions_done(results, label)

    if done >= NUM_QUESTIONS:
        print(f"\n✅ {label}: already complete ({done}/{NUM_QUESTIONS}) - skipping.")
        return True

    print(f"\n{'='*70}\nModel: {label}\n{'='*70}")
    if done > 0:
        print(f"🔄 Resuming from question {done+1}/{NUM_QUESTIONS}")

    kwargs = {"llm_provider": model_cfg["provider"]}
    if model_cfg["groq_model"]:
        kwargs["groq_model"] = model_cfg["groq_model"]
    if model_cfg.get("groq_kwargs"):
        kwargs["groq_kwargs"] = model_cfg["groq_kwargs"]

    print("Initializing agents...")
    agents = {
        "cardiology":  CardiologyAgent(KB["cardiology"], **kwargs),
        "diabetology": DiabetologyAgent(KB["diabetology"], **kwargs),
        "nephrology":  NephrologyAgent(KB["nephrology"], **kwargs),
        "pharmacology":PharmacologyAgent(KB["pharmacology"], **kwargs),
    }
    print("Agents ready.\n")

    if label not in results:
        results[label] = new_model_record()

    for i, q in enumerate(questions):
        if i < results[label]["questions_done"]:
            continue

        correct = get_correct_answer(q)
        qtext = format_question(q)
        print(f"  [{i+1}/{NUM_QUESTIONS}] {q['question'][:60]}...")

        # Collect all 4 agents' answers FIRST; only record if every call succeeded.
        collected = {}
        failed = False
        for specialty in SPECIALTIES:
            agent = agents[specialty]
            t0 = time.time()
            result = analyze_with_retry(agent, qtext)
            elapsed = round(time.time() - t0, 2)
            if result["analysis"].strip().lower().startswith("unavailable"):
                failed = True
                break
            collected[specialty] = (result, elapsed)
            time.sleep(model_cfg.get("pace", 2))  # pacing - respects per-minute limits

        if failed:
            print(f"\n⚠️  {label}: an agent call failed (daily token limit, empty billing "
                  f"balance, or network). Question {i+1} NOT recorded.")
            print(f"   Progress saved: {results[label]['questions_done']}/{NUM_QUESTIONS} "
                  f"questions complete. Rerun later (or tomorrow) to resume.")
            save_progress(results)
            return False

        votes = {}
        for specialty, (result, elapsed) in collected.items():
            ans = extract_answer(result["analysis"])
            is_correct = (ans == correct)
            rec = results[label][specialty]
            rec["total"] += 1
            rec["total_time"] = round(rec["total_time"] + elapsed, 2)
            if is_correct:
                rec["correct"] += 1
            if ans == "X":
                rec["unparsed"] = rec.get("unparsed", 0) + 1
                samples = rec.setdefault("unparsed_samples", [])
                if len(samples) < 3:  # keep a few raw replies so the cause can be diagnosed
                    samples.append(_redact(result["analysis"])[:200])
                print(f"    [WARN] {specialty}: could not parse an answer letter (counted wrong, logged in 'unparsed')")
            if ans != "X":
                votes[ans] = votes.get(ans, 0) + result.get("confidence", 0.0)
            print(f"    {specialty:<14} -> {ans} ({'✓' if is_correct else '✗'}) [{elapsed}s]")

        vote_ans = max(votes, key=votes.get) if votes else "X"
        results[label]["ensemble_vote"]["total"] += 1
        if vote_ans == correct:
            results[label]["ensemble_vote"]["correct"] += 1
        print(f"    Correct: {correct} | Vote: {vote_ans}{'✓' if vote_ans==correct else '✗'}")

        results[label]["questions_done"] = i + 1
        save_progress(results)  # checkpoint after every fully-recorded question

    print(f"\n✅ {label} complete.")
    return True


def run_comparison():
    print("\n" + "=" * 70)
    print("MARC-Clinical Multi-Model Comparison")
    print(f"{len(MODELS)} backbones x {NUM_QUESTIONS} MedQA questions x 4 specialist agents")
    print("=" * 70)

    from datasets import load_dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    questions = list(dataset)[:NUM_QUESTIONS]
    print(f"Loaded {len(questions)} MedQA questions (same sample used for every model).")

    results = load_progress()
    all_complete = True

    for model_cfg in MODELS:
        finished = run_one_model(model_cfg, questions, results)
        if not finished:
            all_complete = False  # paused; continue on to the next model

    print("\n" + "-" * 70)
    print("PROGRESS SUMMARY")
    for m in MODELS:
        print(f"  {m['label']:<20} {questions_done(results, m['label'])}/{NUM_QUESTIONS} questions")
    print("-" * 70)

    if all_complete:
        print_final_table(results)
    else:
        print("Some models are not finished yet. Rerun this same command later; it resumes "
              "where it stopped. The final table prints once all models are complete.")
    return results


def print_final_table(results):
    print("\n" + "=" * 90)
    print("FINAL: MULTI-MODEL COMPARISON - Per-Agent Accuracy")
    print("=" * 90)
    agent_order = SPECIALTIES + ["ensemble_vote"]
    labels = [m["label"] for m in MODELS if m["label"] in results]

    header = f"{'Agent':<16}" + "".join(f"{l:<20}" for l in labels)
    print(header)
    print("-" * len(header))
    for sp in agent_order:
        row = f"{sp.replace('_',' ').capitalize():<16}"
        for l in labels:
            d = results.get(l, {}).get(sp, {"correct": 0, "total": 0})
            acc = round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0.0
            row += f"{acc}%".ljust(20)
        print(row)

    print("\nAvg response time (seconds/question):")
    for sp in SPECIALTIES:
        row = f"{sp.capitalize():<16}"
        for l in labels:
            d = results.get(l, {}).get(sp, {"total_time": 0, "total": 0})
            avg_t = round(d["total_time"] / d["total"], 2) if d["total"] else 0.0
            row += f"{avg_t}s".ljust(20)
        print(row)
    print("=" * 90)

    with open("evaluation/model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved -> evaluation/model_comparison_results.json")


if __name__ == "__main__":
    run_comparison()
