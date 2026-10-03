"""
MedQA single-agent BASELINE re-run (clean).

Why this exists: evaluation/medqa_results.json was produced on 2026-09-01, before the
"Unavailable -> A" parsing fix (2026-09-15). In that file the baseline answered "A" on
26 of 50 questions although only 11 correct answers are "A", so failed baseline calls
were very likely scored as the letter A. The MARC column looks unaffected (its answer
distribution matches the true answer distribution), but the baseline (56.0%) and the
+40 point gap must not be cited until this re-run replaces them.

This script re-runs ONLY the baseline (Cardiology agent, Gemini 3.6 Flash, same 50
MedQA test questions, same prompt as medqa_benchmark.py). It never overwrites
medqa_results.json.

Quota-safe: if a call fails, the question is NOT recorded, progress is saved, and the
script stops so you can rerun later. A failed call is never scored as a wrong answer.

Run:  python evaluation/medqa_baseline_rerun.py
Output: evaluation/medqa_baseline_rerun.json  (progress + final numbers)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.medqa_benchmark import (  # noqa: E402  (also sets cwd to project root)
    KB, format_question, get_correct_answer, extract_answer, analyze_with_retry,
)
from agents.cardiology_agent import CardiologyAgent  # noqa: E402

# Gemini key: uses GOOGLE_API_KEY (your funded key). GOOGLE_API_KEY2 only if MARC_USE_KEY2=1.
if os.getenv("MARC_USE_KEY2") == "1" and os.getenv("GOOGLE_API_KEY2"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY2"]
    print("Using GOOGLE_API_KEY2 for Gemini calls.")

NUM_QUESTIONS = 50
OUT_PATH = "evaluation/medqa_baseline_rerun.json"
OLD_RESULTS = "evaluation/medqa_results.json"


def load_out():
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"baseline_details": []}


def save_out(out):
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)


def main():
    from datasets import load_dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    questions = list(dataset)[:NUM_QUESTIONS]

    out = load_out()
    done = {d["question_id"]: d for d in out["baseline_details"]}
    if done:
        print(f"Resuming: {len(done)}/{NUM_QUESTIONS} already recorded.")

    agent = CardiologyAgent(KB["cardiology"])  # same single-agent baseline as the benchmark

    for i, q in enumerate(questions):
        qid = i + 1
        if qid in done:
            continue
        correct = get_correct_answer(q)
        print(f"[{qid}/{NUM_QUESTIONS}] {q['question'][:60]}...")
        time.sleep(2)
        result = analyze_with_retry(agent, format_question(q))
        if result["analysis"].strip().lower().startswith("unavailable"):
            print(f"\nCall failed on question {qid}: NOT recorded. "
                  f"{len(done)}/{NUM_QUESTIONS} saved. Fix quota/billing and rerun.")
            save_out(out)
            return
        ans = extract_answer(result["analysis"])
        rec = {"question_id": qid, "correct": correct, "baseline_answer": ans,
               "baseline_correct": ans == correct}
        if ans == "X":
            rec["raw_head"] = result["analysis"][:120]
            print("   [WARN] could not parse a letter (counted wrong; raw text saved)")
        done[qid] = rec
        out["baseline_details"] = [done[k] for k in sorted(done)]
        save_out(out)
        print(f"   Correct:{correct} Base:{ans}{'✓' if ans == correct else '✗'}")

    n_correct = sum(1 for d in done.values() if d["baseline_correct"])
    out["num_questions"] = NUM_QUESTIONS
    out["baseline_correct"] = n_correct
    out["baseline_accuracy"] = round(n_correct / NUM_QUESTIONS * 100, 1)
    out["unparsed"] = sum(1 for d in done.values() if d["baseline_answer"] == "X")
    save_out(out)

    print("\n" + "=" * 60)
    print(f"Clean baseline: {out['baseline_accuracy']}% ({n_correct}/{NUM_QUESTIONS}), "
          f"unparsed={out['unparsed']}")
    if os.path.exists(OLD_RESULTS):
        with open(OLD_RESULTS) as f:
            old = json.load(f)
        print(f"Old (pre-fix) baseline: {old['baseline_accuracy']}%  |  "
              f"MARC: {old['marc_accuracy']}%")
        print(f"Corrected MARC-baseline gap: "
              f"{round(old['marc_accuracy'] - out['baseline_accuracy'], 1)} points")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
