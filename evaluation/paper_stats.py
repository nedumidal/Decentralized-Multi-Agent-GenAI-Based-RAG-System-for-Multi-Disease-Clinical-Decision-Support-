"""
Paper statistics for MARC-Clinical.

Reads the saved benchmark result files and prints, for each dataset:
  - MARC and baseline accuracy with 95% Wilson confidence intervals
  - exact McNemar test (paired) for MARC vs baseline
Also prints the multi-model comparison table if evaluation/model_comparison_results.json exists.
Writes everything to evaluation/paper_stats.json.

Run:  python evaluation/paper_stats.py
"""
import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))

# Only cite a number here once you trust the file (see medqa_baseline_rerun.py).
BENCHMARKS = [
    ("MedQA",     "medqa_results.json",     94.0),
    ("PubMedQA",  "pubmedqa_results.json",  88.0),
    ("MedBullets","medbullets_results.json",84.0),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - a) / d * 100, 1), round((c + a) / d * 100, 1))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    out = {"benchmarks": {}, "model_comparison": None}
    print("=" * 84)
    print(f"{'Dataset':<11}{'MARC':<22}{'Baseline':<22}{'Gap':<7}{'McNemar p':<11}{'Base paper'}")
    print("=" * 84)
    for label, fname, base_paper in BENCHMARKS:
        d = load(fname)
        if d is None:
            print(f"{label:<11}(missing {fname})")
            continue
        rows = d["detailed_results"]
        if label == "MedQA":
            # The baseline column in medqa_results.json predates the parsing fix (it scored failed
            # calls as 'A'). Use the clean re-run instead when it is available.
            rr = load("medqa_baseline_rerun.json")
            if rr and len(rr.get("baseline_details", [])) == len(rows):
                for r, b in zip(rows, rr["baseline_details"]):
                    if r.get("correct") != b.get("correct"):
                        sys.exit("MedQA baseline re-run does not match the question set - aborting.")
                    r["baseline_correct"] = b["baseline_correct"]
                    r["baseline_answer"] = b["baseline_answer"]
                print("[MedQA] using the clean baseline re-run (medqa_baseline_rerun.json)")
            else:
                print("[MedQA] WARNING: old baseline column in use - it is unreliable. Run medqa_baseline_rerun.py")
        n = len(rows)
        mk = sum(1 for r in rows if r["marc_correct"])
        bk = sum(1 for r in rows if r["baseline_correct"])
        only_m = sum(1 for r in rows if r["marc_correct"] and not r["baseline_correct"])
        only_b = sum(1 for r in rows if r["baseline_correct"] and not r["marc_correct"])
        p = mcnemar_exact(only_m, only_b)
        mci, bci = wilson(mk, n), wilson(bk, n)
        gap = round((mk - bk) / n * 100, 1)
        print(f"{label:<11}{f'{mk/n*100:.1f}% [{mci[0]}-{mci[1]}]':<22}"
              f"{f'{bk/n*100:.1f}% [{bci[0]}-{bci[1]}]':<22}{gap:<7}{p:<11}{base_paper}%")
        out["benchmarks"][label] = {
            "n": n, "marc_correct": mk, "marc_acc": round(mk / n * 100, 1), "marc_ci95": mci,
            "baseline_correct": bk, "baseline_acc": round(bk / n * 100, 1), "baseline_ci95": bci,
            "gap_points": gap, "marc_only_correct": only_m, "baseline_only_correct": only_b,
            "mcnemar_exact_p": p, "base_paper_reported": base_paper,
        }
    print("=" * 84)
    print("Note: n=50 per dataset -> 95% intervals are roughly +/-10 points wide.")

    mc = load("model_comparison_results.json") or load("model_comparison_progress.json")
    if mc:
        out["model_comparison"] = mc
        specs = ["cardiology", "nephrology", "diabetology", "pharmacology", "ensemble_vote"]
        # Only fully finished models go in the paper tables; partial runs are listed below.
        labels = [l for l in mc if mc[l].get("questions_done", 0) >= 20]
        partial = {l: mc[l].get("questions_done", 0) for l in mc if l not in labels}
        if partial:
            print("\nNot included (incomplete): " + ", ".join(f"{l} {n}/20" for l, n in partial.items()))
        print("\nMulti-model comparison (per-agent accuracy, questions completed in brackets)")
        print(f"{'Agent':<16}" + "".join(f"{l} [{mc[l].get('questions_done', 0)}]".ljust(28) for l in labels))
        for sp in specs:
            row = f"{sp.replace('_', ' ').capitalize():<16}"
            for l in labels:
                dd = mc[l].get(sp, {"correct": 0, "total": 0})
                if dd["total"]:
                    k, t = dd["correct"], dd["total"]
                    lo, hi = wilson(k, t)
                    row += f"{k / t * 100:.0f}% [{lo:.0f}-{hi:.0f}]".ljust(28)
                else:
                    row += "n/a".ljust(28)
            print(row)
        print("\nAvg response time (s/question)")
        for sp in specs[:4]:
            row = f"{sp.capitalize():<16}"
            for l in labels:
                dd = mc[l].get(sp, {"total_time": 0, "total": 0})
                row += (f"{dd['total_time'] / dd['total']:.2f}s" if dd["total"] else "n/a").ljust(28)
            print(row)
    else:
        print("\n(no model_comparison results yet - run evaluation/model_comparison.py)")

    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> evaluation/paper_stats.json")


if __name__ == "__main__":
    main()
