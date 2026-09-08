"""
AFL — Adaptive Feedback Loop
Novel Contribution #5 of MARC-Clinical Framework
FIXED: uses _get_rag_query() instead of removed _get_focused_query()
"""

import time
import re
from datetime import datetime


class AFL:
    def __init__(self, confidence_threshold: float = 0.65, max_iterations: int = 3):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.iteration_log = []
        self.total_refinements = 0
        self.queries_expanded = []

    def _extract_case_specifics(self, case_text: str) -> str:
        patterns = [
            r"eGFR\s*[:=]?\s*\d+(?:\.\d+)?",
            r"EF\s*[:=]?\s*\d+%?",
            r"HbA1c\s*[:=]?\s*\d+(?:\.\d+)?%?",
            r"BNP\s*[:=]?\s*\d+",
            r"[Cc]reatinine\s*[:=]?\s*\d+(?:\.\d+)?",
            r"NYHA\s*(?:Class\s*)?[IVX]+",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, case_text))
        seen = set()
        unique = []
        for f in found:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return " ".join(unique)

    def _expand_query(self, original_query: str, iteration: int,
                      specialty: str, case_specifics: str = "") -> str:
        expansions = {
            "cardiology": [
                "heart failure cardiomyopathy systolic dysfunction treatment guidelines",
                "cardiac failure ventricular dysfunction medication therapy protocol",
                "HF HFrEF HFpEF treatment beta blocker ACE inhibitor diuretic",
            ],
            "nephrology": [
                "chronic kidney disease renal failure management clinical guidelines",
                "CKD GFR creatinine proteinuria treatment nephrology protocol",
                "kidney disease progression prevention ACE ARB SGLT2",
            ],
            "diabetology": [
                "diabetes mellitus type 2 treatment glycemic control guidelines",
                "T2DM insulin SGLT2 GLP1 HbA1c management protocol",
                "diabetic treatment comorbidity heart failure kidney disease",
            ],
            "pharmacology": [
                "drug therapy medication dosing renal impairment guidelines",
                "pharmacokinetics kidney failure dose adjustment drug interaction",
                "WHO essential medicines dosing protocol clinical recommendation",
            ],
        }
        options = expansions.get(specialty, [original_query])
        idx = min(iteration - 1, len(options) - 1)
        expanded = options[idx]
        if case_specifics:
            expanded = f"{expanded} {case_specifics}"
        self.queries_expanded.append({
            "specialty": specialty, "original": original_query,
            "expanded": expanded, "iteration": iteration,
        })
        return expanded

    def needs_feedback(self, results: dict, dcwo_consensus: dict) -> bool:
        avg_confidence = (
            sum(r["confidence"] for r in results.values()) / len(results)
            if results else 0
        )
        consensus_reached = dcwo_consensus.get("consensus_reached", False)
        return avg_confidence < self.confidence_threshold or not consensus_reached

    def run_feedback_loop(self, results: dict, agents: dict,
                          dcwo_report: dict, sral, iakb, scdp, dcwo,
                          original_case: str) -> dict:
        if not self.needs_feedback(results, dcwo_report):
            print("[AFL] Confidence above threshold — no feedback needed.")
            return results

        print(f"\n[AFL] Confidence below {self.confidence_threshold} "
              f"— initiating adaptive feedback loop...")

        current_results = results.copy()
        best_results = results.copy()
        best_confidence = (
            sum(r["confidence"] for r in results.values()) / len(results)
            if results else 0
        )
        case_specifics = self._extract_case_specifics(original_case)

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[AFL] ── Iteration {iteration}/{self.max_iterations} ──")
            iter_start = time.time()
            improved_results = {}

            for specialty, agent_obj in agents.items():
                old_result = current_results.get(specialty, {})
                old_conf = old_result.get("confidence", 0)

                if old_conf >= self.confidence_threshold:
                    improved_results[specialty] = old_result
                    print(f"[AFL]   {specialty}: confidence {old_conf} OK — skip")
                    continue

                # ── FIX: use _get_rag_query() — agents no longer have _get_focused_query()
                if hasattr(agent_obj, "_get_rag_query"):
                    base_query = agent_obj._get_rag_query(original_case)
                elif hasattr(agent_obj, "_get_pubmed_query"):
                    base_query = agent_obj._get_pubmed_query(original_case)
                else:
                    base_query = f"{specialty} clinical guidelines treatment"

                expanded_query = self._expand_query(
                    base_query, iteration, specialty, case_specifics
                )
                print(f"[AFL]   {specialty}: re-querying...")
                print(f"[AFL]   Query: {expanded_query[:70]}...")

                try:
                    agent_obj.retriever.invoke(expanded_query)
                    prior = iakb.format_findings_for_context(agent_obj.name)
                    new_result = agent_obj.analyze(original_case + prior)
                    sral.register_retrieval(agent_obj.name, new_result["retrieved_docs"])
                    new_conf = new_result["confidence"]
                    print(f"[AFL]   {specialty}: {old_conf} → {new_conf}")
                    improved_results[specialty] = new_result
                except Exception as e:
                    print(f"[AFL]   {specialty}: re-query failed — {e}")
                    improved_results[specialty] = old_result

            new_conflicts = scdp.detect_and_resolve(improved_results)
            new_consensus = dcwo.run_consensus(improved_results, new_conflicts)

            new_avg_conf = (
                sum(r["confidence"] for r in improved_results.values()) / len(improved_results)
                if improved_results else 0
            )
            iter_time = round(time.time() - iter_start, 1)

            log_entry = {
                "iteration": iteration,
                "avg_confidence": round(new_avg_conf, 3),
                "conflicts": len(new_conflicts),
                "consensus": new_consensus.get("consensus_reached", False),
                "time_seconds": iter_time,
                "timestamp": datetime.now().isoformat(),
            }
            self.iteration_log.append(log_entry)
            self.total_refinements += 1

            print(f"[AFL]   Round {iteration}: avg_conf={new_avg_conf:.3f} "
                  f"conflicts={len(new_conflicts)} time={iter_time}s")

            if new_avg_conf > best_confidence:
                best_confidence = new_avg_conf
                best_results = improved_results.copy()

            current_results = improved_results

            if new_avg_conf >= self.confidence_threshold:
                print(f"\n[AFL] ✅ Threshold reached at iteration {iteration}!")
                break

        print(f"\n[AFL] Feedback loop complete. Best confidence: {best_confidence:.3f}")
        return best_results

    def get_report(self) -> dict:
        return {
            "total_refinements": self.total_refinements,
            "max_iterations_allowed": self.max_iterations,
            "confidence_threshold": self.confidence_threshold,
            "queries_expanded": len(self.queries_expanded),
            "iteration_log": self.iteration_log,
            "accuracy_progression": [
                {"iteration": l["iteration"], "avg_confidence": l["avg_confidence"]}
                for l in self.iteration_log
            ],
        }

    def print_report(self):
        report = self.get_report()
        print("\n" + "="*60)
        print("AFL REPORT — Adaptive Feedback Loop")
        print("="*60)
        print(f"Total refinement rounds : {report['total_refinements']}")
        print(f"Queries expanded        : {report['queries_expanded']}")
        print(f"Confidence threshold    : {report['confidence_threshold']}")
        if report["iteration_log"]:
            print("\nAccuracy progression:")
            for entry in report["iteration_log"]:
                bar = "█" * int(entry["avg_confidence"] * 20)
                print(f"  Round {entry['iteration']}: {bar} {entry['avg_confidence']} ({entry['time_seconds']}s)")
        print("="*60)
