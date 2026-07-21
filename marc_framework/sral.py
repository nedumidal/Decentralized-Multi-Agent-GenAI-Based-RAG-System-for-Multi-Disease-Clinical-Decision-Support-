"""
SRAL — Shared Retrieval Awareness Layer
Novel Contribution #1 of MARC-Clinical Framework

When multiple agents retrieve the same document chunk,
SRAL detects this, logs it, and enables cross-agent
interpretation sharing — preventing isolated contradicting
beliefs from the same source.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional


class SRAL:
    def __init__(self):
        # Central registry: doc_hash -> {agent, page, preview, timestamp}
        self.retrieval_registry = {}
        # Duplication log for metrics
        self.duplication_log = []
        # Cross-agent interpretation store
        self.interpretation_store = {}

    def _generate_doc_hash(self, content: str) -> str:
        """Generate unique hash for a document chunk"""
        return hashlib.md5(content.strip()[:200].encode()).hexdigest()

    def register_retrieval(self, agent_name: str, retrieved_docs: list) -> dict:
        """
        Register what an agent retrieved.
        Returns: dict with duplicates found and shared interpretations
        """
        duplicates_found = []
        new_registrations = []

        for doc in retrieved_docs:
            content = doc.get("content_preview", "")
            doc_id = doc.get("doc_id", "unknown")
            source = doc.get("source", "unknown")
            doc_hash = self._generate_doc_hash(content)

            if doc_hash in self.retrieval_registry:
                # DUPLICATE DETECTED
                original = self.retrieval_registry[doc_hash]
                duplicate_event = {
                    "doc_hash": doc_hash,
                    "original_agent": original["agent"],
                    "duplicate_agent": agent_name,
                    "page": doc_id,
                    "content_preview": content[:100],
                    "timestamp": datetime.now().isoformat(),
                    "shared_interpretation": self.interpretation_store.get(doc_hash)
                }
                duplicates_found.append(duplicate_event)
                self.duplication_log.append(duplicate_event)

                print(f"\n[SRAL] ⚠️  DUPLICATE DETECTED!")
                print(f"       Document (page {doc_id}) retrieved by both:")
                print(f"       → {original['agent']} (first)")
                print(f"       → {agent_name} (now)")
                print(f"       Sharing {original['agent']}'s interpretation...")

            else:
                # NEW RETRIEVAL — register it
                self.retrieval_registry[doc_hash] = {
                    "agent": agent_name,
                    "doc_id": doc_id,
                    "source": source,
                    "content_preview": content[:100],
                    "timestamp": datetime.now().isoformat()
                }
                new_registrations.append(doc_hash)

        return {
            "agent": agent_name,
            "total_retrieved": len(retrieved_docs),
            "new_retrievals": len(new_registrations),
            "duplicates_found": len(duplicates_found),
            "duplicate_details": duplicates_found
        }

    def store_interpretation(self, agent_name: str,
                             retrieved_docs: list, analysis: str):
        """Store agent's interpretation of retrieved docs for sharing"""
        for doc in retrieved_docs:
            content = doc.get("content_preview", "")
            doc_hash = self._generate_doc_hash(content)
            if doc_hash not in self.interpretation_store:
                self.interpretation_store[doc_hash] = {
                    "agent": agent_name,
                    "interpretation_summary": analysis[:300],
                    "timestamp": datetime.now().isoformat()
                }

    def get_deduplication_rate(self) -> float:
        """
        Novel Metric: Retrieval Deduplication Rate
        % of retrievals that were duplicates across agents
        """
        total = len(self.retrieval_registry) + len(self.duplication_log)
        if total == 0:
            return 0.0
        return round((len(self.duplication_log) / total) * 100, 2)

    def get_report(self) -> dict:
        """Generate SRAL performance report for evaluation"""
        return {
            "total_unique_documents": len(self.retrieval_registry),
            "total_duplicate_retrievals": len(self.duplication_log),
            "deduplication_rate_percent": self.get_deduplication_rate(),
            "agents_involved": list(set(
                [v["agent"] for v in self.retrieval_registry.values()]
            )),
            "duplicate_pairs": [
                f"{d['original_agent']} ↔ {d['duplicate_agent']}"
                for d in self.duplication_log
            ]
        }

    def print_report(self):
        report = self.get_report()
        print("\n" + "="*60)
        print("SRAL REPORT — Shared Retrieval Awareness Layer")
        print("="*60)
        print(f"Total unique documents retrieved : {report['total_unique_documents']}")
        print(f"Duplicate retrievals detected    : {report['total_duplicate_retrievals']}")
        print(f"Deduplication rate               : {report['deduplication_rate_percent']}%")
        print(f"Agents involved                  : {report['agents_involved']}")
        if report['duplicate_pairs']:
            print(f"Duplicate pairs                  :")
            for pair in report['duplicate_pairs']:
                print(f"  → {pair}")
        print("="*60)