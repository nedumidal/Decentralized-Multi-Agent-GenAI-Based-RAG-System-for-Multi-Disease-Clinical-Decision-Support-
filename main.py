import os
import time
from dotenv import load_dotenv
from agents.cardiology_agent import CardiologyAgent
from agents.diabetology_agent import DiabetologyAgent
from agents.nephrology_agent import NephrologyAgent
from agents.pharmacology_agent import PharmacologyAgent

load_dotenv()

test_case = """
Patient: 58-year-old male
Symptoms: Shortness of breath, leg swelling, fatigue, excessive thirst
Lab Results: BNP 890 pg/mL, EF 35%, Creatinine 1.8 mg/dL, HbA1c 8.2%, eGFR 42
Current medications: None
Question: What is the recommended treatment approach for this patient?
"""

print("\n" + "="*60)
print("MARC-Clinical: Multi-Agent RAG System Starting...")
print("="*60)

agents = {
    "cardiology": CardiologyAgent("knowledge_bases/cardiology/aha_heart_failure.pdf"),
    "diabetology": DiabetologyAgent("knowledge_bases/diabetology/ada_standards_2025.pdf"),
    "nephrology": NephrologyAgent("knowledge_bases/nephrology/kdigo_ckd_2024.pdf"),
    "pharmacology": PharmacologyAgent("knowledge_bases/pharmacology/who_essential_medicines.pdf"),
}

def analyze_with_retry(agent, case, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.analyze(case)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = (attempt + 1) * 15
                print(f"  Server busy. Retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e
    return {
        "agent": agent.name,
        "specialty": agent.specialty,
        "analysis": "Server unavailable after retries. Run again.",
        "retrieved_docs": [],
        "confidence": 0.0
    }

results = {}
for specialty, agent in agents.items():
    print(f"\n--- Running {agent.name} ---")
    results[specialty] = analyze_with_retry(agent, test_case)

print("\n" + "="*60)
print("MARC-Clinical: Combined Multi-Agent Analysis")
print("="*60)

for specialty, result in results.items():
    print(f"\n🏥 {result['agent']} (Confidence: {result['confidence']})")
    print(f"{result['analysis']}")
    print(f"Retrieved {len(result['retrieved_docs'])} chunks from guidelines")
    print("-"*60)

print("\n✅ All 4 agents completed. MARC framework base ready.")
print("Next step: Implement SRAL for retrieval deduplication.")