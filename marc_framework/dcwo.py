"""
DCWO — Decentralized Consensus Without Orchestrator
Novel Contribution #4 of MARC-Clinical Framework

Agents iteratively share and update their beliefs with
each other until convergence — NO central controller.
Inspired by gossip protocol from distributed systems,
applied to LLM agent coordination for the first time.

This replaces the single Data Fusion Agent used in the
base paper (IEEE Access 2025) which is a single point
of failure and bias toward one specialty.
"""

from datetime import datetime


class DCWO:
    def __init__(self, convergence_threshold: float = 0.85,
                 max_rounds: int = 5):
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        self.rounds_log = []
        self.consensus_reached = False
        self.convergence_round = None
        self.final_consensus = {}

    def _calculate_agreement_score(self,
                                   results: dict,
                                   conflicts: list) -> float:
        """
        Calculate how much agents agree with each other.
        Score = 1.0 means full consensus, 0.0 means full conflict.
        """
        total_agents = len(results)
        if total_agents == 0:
            return 0.0

        # Base score from agent confidences
        avg_confidence = sum(
            r["confidence"] for r in results.values()
        ) / total_agents

        # Penalty for each conflict
        conflict_penalty = len(conflicts) * 0.15

        score = max(0.0, avg_confidence - conflict_penalty)
        return round(score, 3)

    def _extract_key_positions(self, results: dict) -> dict:
        """Extract each agent's position on key clinical decisions"""
        positions = {}
        for specialty, result in results.items():
            analysis = result["analysis"].lower()
            positions[result["agent"]] = {
                "sglt2i": "sglt2" in analysis,
                "ace_inhibitor": "ace inhibitor" in analysis or "acei" in analysis,
                "beta_blocker": "beta-blocker" in analysis or "bisoprolol" in analysis,
                "loop_diuretic": "furosemide" in analysis or "loop diuretic" in analysis,
                "glp1": "glp-1" in analysis or "semaglutide" in analysis,
                "mra": "spironolactone" in analysis or "mra" in analysis,
                "confidence": result["confidence"]
            }
        return positions

    def _calculate_position_agreement(self, positions: dict) -> dict:
        """Calculate how many agents agree on each treatment"""
        all_agents = list(positions.values())
        n = len(all_agents)
        if n == 0:
            return {}

        keys = [k for k in all_agents[0].keys() if k != "confidence"]
        agreement = {}

        for key in keys:
            votes = sum(1 for agent in all_agents if agent.get(key, False))
            agreement[key] = round(votes / n, 2)

        return agreement

    def _update_agent_beliefs(self, results: dict,
                              positions: dict,
                              agreement: dict,
                              round_num: int) -> dict:
        """
        Gossip step: agents update their confidence based on
        peer agreement — higher agreement = higher confidence.
        This simulates agents reading each other's positions
        and updating their own beliefs.
        """
        updated = {}
        for specialty, result in results.items():
            agent_name = result["agent"]
            agent_pos = positions.get(agent_name, {})
            old_confidence = result["confidence"]

            # Count how many of this agent's positions
            # are supported by majority of other agents
            supported = 0
            total_positions = 0

            for key, value in agent_pos.items():
                if key == "confidence":
                    continue
                total_positions += 1
                if value and agreement.get(key, 0) >= 0.5:
                    supported += 1
                elif not value and agreement.get(key, 0) < 0.5:
                    supported += 1

            support_ratio = (supported / total_positions
                             if total_positions > 0 else 0.5)

            # Update confidence toward peer agreement
            new_confidence = round(
                (old_confidence * 0.7) + (support_ratio * 0.3), 3
            )

            updated[specialty] = {
                **result,
                "confidence": new_confidence,
                "round": round_num
            }

        return updated

    def run_consensus(self, results: dict,
                      conflicts: list) -> dict:
        """
        Main DCWO algorithm — gossip-protocol-inspired consensus.
        Runs up to max_rounds until convergence threshold is met.
        """
        print(f"\n[DCWO] Starting decentralized consensus...")
        print(f"[DCWO] Agents: {[r['agent'] for r in results.values()]}")
        print(f"[DCWO] Convergence threshold: {self.convergence_threshold}")
        print(f"[DCWO] Max rounds: {self.max_rounds}")

        current_results = results.copy()

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[DCWO] --- Round {round_num} ---")

            # Step 1: Calculate current agreement score
            agreement_score = self._calculate_agreement_score(
                current_results, conflicts
            )

            # Step 2: Extract positions
            positions = self._extract_key_positions(current_results)

            # Step 3: Calculate position-level agreement
            position_agreement = self._calculate_position_agreement(
                positions
            )

            # Step 4: Log this round
            round_log = {
                "round": round_num,
                "agreement_score": agreement_score,
                "agent_confidences": {
                    r["agent"]: r["confidence"]
                    for r in current_results.values()
                },
                "position_agreement": position_agreement
            }
            self.rounds_log.append(round_log)

            print(f"[DCWO] Agreement score: {agreement_score}")
            print(f"[DCWO] Agent confidences: "
                  f"{round_log['agent_confidences']}")
            print(f"[DCWO] Treatment consensus: "
                  f"{position_agreement}")

            # Step 5: Check convergence
            if agreement_score >= self.convergence_threshold:
                self.consensus_reached = True
                self.convergence_round = round_num
                print(f"\n[DCWO] ✅ CONSENSUS REACHED at round {round_num}!")
                print(f"[DCWO] Final agreement score: {agreement_score}")
                break

            # Step 6: Gossip update — agents update beliefs
            current_results = self._update_agent_beliefs(
                current_results, positions,
                position_agreement, round_num
            )
            print(f"[DCWO] Agents updated beliefs — "
                  f"next round starting...")

        if not self.consensus_reached:
            print(f"\n[DCWO] ⚠️  Max rounds reached without full consensus.")
            print(f"[DCWO] Best agreement: "
                  f"{self.rounds_log[-1]['agreement_score']}")

        # Build final consensus output
        self.final_consensus = self._build_final_consensus(
            current_results, position_agreement
        )

        return self.final_consensus

    def _build_final_consensus(self, results: dict,
                                position_agreement: dict) -> dict:
        """Build unified consensus recommendation"""
        consensus_treatments = {
            treatment: agreed
            for treatment, agreed in position_agreement.items()
            if agreed >= 0.5  # Majority agreement
        }

        return {
            "consensus_treatments": consensus_treatments,
            "all_agents_agree_on": [
                t for t, score in position_agreement.items()
                if score == 1.0
            ],
            "majority_agree_on": [
                t for t, score in position_agreement.items()
                if 0.5 <= score < 1.0
            ],
            "convergence_round": self.convergence_round,
            "consensus_reached": self.consensus_reached,
            "final_agent_confidences": {
                r["agent"]: r["confidence"]
                for r in results.values()
            }
        }

    def get_convergence_report(self) -> dict:
        """Novel Metric: Consensus Convergence Rounds"""
        return {
            "consensus_reached": self.consensus_reached,
            "convergence_round": self.convergence_round,
            "total_rounds_run": len(self.rounds_log),
            "max_rounds_allowed": self.max_rounds,
            "round_by_round_scores": [
                {"round": r["round"],
                 "agreement_score": r["agreement_score"]}
                for r in self.rounds_log
            ],
            "final_consensus": self.final_consensus
        }

    def print_report(self):
        report = self.get_convergence_report()
        print("\n" + "="*60)
        print("DCWO REPORT — Decentralized Consensus Without Orchestrator")
        print("="*60)
        print(f"Consensus reached     : {report['consensus_reached']}")
        print(f"Convergence round     : {report['convergence_round']}")
        print(f"Total rounds run      : {report['total_rounds_run']}")
        print(f"\nRound-by-round scores:")
        for r in report['round_by_round_scores']:
            bar = "█" * int(r['agreement_score'] * 20)
            print(f"  Round {r['round']}: {bar} {r['agreement_score']}")

        if report['final_consensus']:
            fc = report['final_consensus']
            print(f"\nAll agents agree on   : {fc.get('all_agents_agree_on', [])}")
            print(f"Majority agree on     : {fc.get('majority_agree_on', [])}")
            print(f"\nFinal confidences:")
            for agent, conf in fc.get('final_agent_confidences', {}).items():
                print(f"  {agent}: {conf}")
        print("="*60)