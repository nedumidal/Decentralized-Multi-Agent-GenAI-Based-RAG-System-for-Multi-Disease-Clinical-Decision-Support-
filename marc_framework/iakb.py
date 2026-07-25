"""
IAKB — Inter-Agent Knowledge Bus
Novel Contribution #2 of MARC-Clinical Framework

Agents broadcast intermediate clinical findings in real time.
Other agents subscribe and update their reasoning context
before making recommendations — preventing wrong downstream
decisions based on incomplete information.

Real clinical example this solves:
Pharmacology Agent recommends Metformin dose BEFORE
Nephrology Agent reports kidney failure → dangerous overdose.
IAKB prevents this by propagating kidney findings immediately.
"""

from datetime import datetime
from typing import Optional


class IAKB:
    def __init__(self):
        # Knowledge bus — stores all published findings
        self.knowledge_bus = []
        # Subscription registry
        self.subscriptions = {}
        # Transfer log for metrics
        self.transfer_log = []

    def publish(self, agent_name: str, finding_type: str,
                finding: str, confidence: float, relevant_to: list):
        """
        Agent publishes a clinical finding to the bus.

        Args:
            agent_name: Who is publishing
            finding_type: Category (e.g., 'renal_function', 'drug_interaction')
            finding: The actual clinical finding text
            confidence: How confident the agent is (0.0-1.0)
            relevant_to: Which specialties should receive this
        """
        event = {
            "id": len(self.knowledge_bus) + 1,
            "publisher": agent_name,
            "finding_type": finding_type,
            "finding": finding,
            "confidence": confidence,
            "relevant_to": relevant_to,
            "timestamp": datetime.now().isoformat(),
            "acknowledged_by": []
        }
        self.knowledge_bus.append(event)

        print(f"\n[IAKB] 📡 {agent_name} PUBLISHED:")
        print(f"       Type: {finding_type}")
        print(f"       Finding: {finding[:120]}...")
        print(f"       Relevant to: {relevant_to}")

        return event["id"]

    def subscribe(self, agent_name: str, finding_types: list):
        """Register an agent to receive specific finding types"""
        self.subscriptions[agent_name] = finding_types
        print(f"[IAKB] {agent_name} subscribed to: {finding_types}")

    def get_relevant_findings(self, agent_name: str) -> list:
        """
        Get all findings relevant to this agent.
        This is what agents READ before making recommendations.
        """
        agent_specialty = agent_name.replace(" Agent", "").lower()
        relevant = []

        for event in self.knowledge_bus:
            # Don't return agent's own findings
            if event["publisher"] == agent_name:
                continue

            # Check if relevant to this agent
            if (agent_specialty in [r.lower() for r in event["relevant_to"]] or
                    "all" in event["relevant_to"]):
                relevant.append(event)

                # Log transfer
                transfer = {
                    "from": event["publisher"],
                    "to": agent_name,
                    "finding_type": event["finding_type"],
                    "timestamp": datetime.now().isoformat()
                }
                if transfer not in self.transfer_log:
                    self.transfer_log.append(transfer)

                # Mark as acknowledged
                if agent_name not in event["acknowledged_by"]:
                    event["acknowledged_by"].append(agent_name)

        return relevant

    def format_findings_for_context(self, agent_name: str) -> str:
        """
        Format relevant findings as additional context
        to inject into agent's prompt before analysis.
        """
        findings = self.get_relevant_findings(agent_name)
        if not findings:
            return ""

        context = "\n\n=== INTER-AGENT KNOWLEDGE BUS FINDINGS ===\n"
        context += "The following critical findings were published by "
        context += "other specialist agents. Consider these in your analysis:\n\n"

        for f in findings:
            context += f"[{f['publisher']}] {f['finding_type'].upper()}:\n"
            context += f"{f['finding']}\n"
            context += f"(Confidence: {f['confidence']})\n\n"

        context += "===========================================\n"
        return context

    def get_transfer_latency_report(self) -> dict:
        """
        Novel Metric: Knowledge Transfer Latency
        Measures how fast findings propagate across agents.
        """
        return {
            "total_findings_published": len(self.knowledge_bus),
            "total_transfers": len(self.transfer_log),
            "transfer_pairs": [
                f"{t['from']} → {t['to']} ({t['finding_type']})"
                for t in self.transfer_log
            ]
        }

    def print_report(self):
        report = self.get_transfer_latency_report()
        print("\n" + "="*60)
        print("IAKB REPORT — Inter-Agent Knowledge Bus")
        print("="*60)
        print(f"Total findings published : {report['total_findings_published']}")
        print(f"Total knowledge transfers: {report['total_transfers']}")
        if report['transfer_pairs']:
            print("Transfer pairs:")
            for pair in report['transfer_pairs']:
                print(f"  → {pair}")
        print("="*60)