"""
Web Evidence Scanner - mirrors the base paper's (Ogdu et al., IEEE Access 2025)
Evidence-Based Scanner Agent, which used Serper to query PubMed/Google Scholar/NICE
in real time. Their own numbers showed web evidence ALONE (70.0%) beat static RAG
ALONE (65.0%) on MedQA, and fusing both reached 94.7% - so live web evidence is one
of their highest-leverage components, and one MARC-Clinical did not have until now.

Uses `ddgs` (free, keyless DuckDuckGo-backed search) instead of Serper to avoid
adding a second paid API dependency on top of the Gemini quota you're already
managing. Swap in Serper/SerpAPI later if you want closer parity with the paper's
exact evidence sources (PubMed/Scholar/NICE) - the fusion step doesn't care which
backend supplied the evidence, only that it's fresh.
"""

def get_web_evidence(query: str, max_results: int = 3) -> str:
    """
    Fetch live web evidence snippets for a clinical question.
    Returns empty string on any failure (network, blocking, no results) so a
    web outage never breaks the pipeline - it just falls back to RAG-only,
    same as the paper's own degraded-mode behavior implies.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return ""

    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                f"{query} clinical guideline evidence",
                max_results=max_results,
                safesearch="moderate",
            )
        if not results:
            return ""

        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")[:300]
            source = r.get("href", "")
            if body:
                snippets.append(f"[{title}] {body} (source: {source})")

        return "\n".join(snippets) if snippets else ""
    except Exception as e:
        print(f"  [WebEvidence] search failed (non-fatal, continuing without it): {e}")
        return ""
