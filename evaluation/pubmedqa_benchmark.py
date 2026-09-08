"""
PubMedQA Benchmark Evaluation for MARC-Clinical

PubMedQA's standard task: given a research question + a PubMed abstract as context,
answer yes/no/maybe. This is NOT multiple choice like MedQA - it's a 3-way
classification. The base paper (Section III.C) scored PubMedQA/MedBullets via a
cosine-similarity threshold (theta=0.80) against free-text answers; here we score
via exact match on the yes/no/maybe label instead, since that IS PubMedQA's actual
official evaluation protocol (Jin et al. 2019) and is more rigorous/reproducible
than a similarity threshold. Note this scoring difference in your methodology
section - it's a legitimate deviation, not an error, but it should be disclosed.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

from agents.cardiology_agent import CardiologyAgent
from agents.diabetology_agent import DiabetologyAgent
from agents.nephrology_agent import NephrologyAgent
from agents.pharmacology_agent import PharmacologyAgent
from marc_framework.sral import SRAL
from marc_framework.iakb import IAKB
from marc_framework.scdp import SCDP
from marc_framework.dcwo import DCWO
from marc_framework.afl import AFL
from marc_framework.web_evidence import get_web_evidence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

KB = {
    "cardiology":  os.path.join(ROOT, "knowledge_bases/cardiology/aha_heart_failure.pdf"),
    "diabetology": os.path.join(ROOT, "knowledge_bases/diabetology/ada_standards_2025.pdf"),
    "nephrology":  os.path.join(ROOT, "knowledge_bases/nephrology/kdigo_ckd_2024.pdf"),
    "pharmacology":os.path.join(ROOT, "knowledge_bases/pharmacology/who_essential_medicines.pdf"),
}

def format_question(q):
    """PubMedQA gives a research question plus an abstract's contexts."""
    context = q.get("context", {})
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        ctx_str = "\n".join(contexts) if contexts else "(no context abstract provided)"
    else:
        ctx_str = str(context)
    return (f"Biomedical Research Question:\n{q['question']}\n\n"
            f"Abstract Context:\n{ctx_str[:2000]}\n\n"
            "Based ONLY on the abstract context above, answer yes, no, or maybe. "
            "Start your response with exactly one word: yes, no, or maybe.")

def extract_answer(text):
    if not text:
        return "unclear"
    t = text.strip().lower()
    for word in ["yes", "no", "maybe"]:
        if t.startswith(word):
            return word
    for marker in ["answer: ", "answer is ", "final answer: "]:
        if marker in t:
            idx = t.index(marker) + len(marker)
            rest = t[idx:idx+10]
            for word in ["yes", "no", "maybe"]:
                if rest.startswith(word):
                    return word
    for word in ["yes", "no", "maybe"]:
        if f" {word} " in f" {t[:200]} " or f" {word}." in t[:200]:
            return word
    return "unclear"

def get_correct_answer(q):
    val = str(q.get("final_decision", "")).strip().lower()
    return val if val in ["yes", "no", "maybe"] else "unclear"

def analyze_with_retry(agent, case, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return agent.analyze(case)
        except Exception as e:
            last_error = e
            es = str(e)
            print(f"  [{agent.name}] error: {es[:300]}")
            if "PERMISSION_DENIED" in es or "403" in es:
                print(f"  [{agent.name}] account/billing-level block — retrying won't help. "
                      f"Check your Google Cloud project's billing status. Skipping retries.")
                break
            wait = 20 if "503" in es or "UNAVAILABLE" in es else (40 if "429" in es else 10)
            if attempt == max_retries-1:
                break
            print(f"  Retry {attempt+1} in {wait}s...")
            time.sleep(wait)
    print(f"  [{agent.name}] giving up. Last error: {str(last_error)[:300]}")
    return {"agent":agent.name,"specialty":agent.specialty,
            "analysis":"Unavailable.","retrieved_docs":[],"confidence":0.0}

_fusion_llm = None
def get_fusion_llm():
    global _fusion_llm
    if _fusion_llm is None:
        _fusion_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), timeout=90)
    return _fusion_llm

FUSION_PROMPT = PromptTemplate.from_template("""You are the Clinical Data Fusion arbiter for a multi-agent medical panel.
Four specialist agents (Cardiology, Nephrology, Diabetology, Pharmacology) independently analyzed the
same biomedical research question below, using a PubMed abstract as context. Your job is to weigh their
reasoning, resolve any disagreement, and output the single best final answer: yes, no, or maybe.

Question:
{question}

--- Cardiology Agent (self-reported confidence {c_conf}) ---
{c_analysis}

--- Nephrology Agent (self-reported confidence {n_conf}) ---
{n_analysis}

--- Diabetology Agent (self-reported confidence {d_conf}) ---
{d_analysis}

--- Pharmacology Agent (self-reported confidence {p_conf}) ---
{p_analysis}

--- Live Web Evidence (independent of the RAG knowledge bases above) ---
{web_evidence}

Conflicts flagged by the conflict-detection layer: {conflicts}

Instructions:
- The correct answer must be grounded in the abstract context given in the question, not general
  knowledge - PubMedQA specifically tests whether the abstract supports, contradicts, or is
  inconclusive about the research question.
- Not every specialist's domain is relevant to every question; down-weight agents reasoning outside
  their expertise even if their stated confidence is high.
- Respond with ONLY one word on the first line: yes, no, or maybe. Follow with a one-sentence
  justification on the next line.""")

def synthesize_final_answer(question_text, results, conflicts, web_evidence=""):
    llm = get_fusion_llm()
    chain = FUSION_PROMPT | llm | StrOutputParser()
    conflict_summary = "; ".join(str(c) for c in conflicts) if conflicts else "None detected"
    out = chain.invoke({
        "question": question_text,
        "c_analysis": results["cardiology"]["analysis"][:600],
        "c_conf": results["cardiology"]["confidence"],
        "n_analysis": results["nephrology"]["analysis"][:600],
        "n_conf": results["nephrology"]["confidence"],
        "d_analysis": results["diabetology"]["analysis"][:600],
        "d_conf": results["diabetology"]["confidence"],
        "p_analysis": results["pharmacology"]["analysis"][:600],
        "p_conf": results["pharmacology"]["confidence"],
        "web_evidence": web_evidence if web_evidence else "(no web evidence retrieved for this question)",
        "conflicts": conflict_summary,
    })
    return extract_answer(out)

def get_marc_answer(question_text, agents, sral, iakb, scdp, dcwo):
    results = {}
    order = [
        ("cardiology",  agents["cardiology"],  "cardiac_function", ["nephrology","diabetology","pharmacology"]),
        ("nephrology",  agents["nephrology"],  "renal_function",   ["cardiology","diabetology","pharmacology"]),
        ("diabetology", agents["diabetology"], "glucose_control",  ["cardiology","nephrology","pharmacology"]),
        ("pharmacology",agents["pharmacology"],"drug_safety",      ["cardiology","nephrology","diabetology"]),
    ]
    for specialty, agent, ftype, notify in order:
        prior = iakb.format_findings_for_context(agent.name)
        result = analyze_with_retry(agent, question_text+prior)
        sral.register_retrieval(agent.name, result["retrieved_docs"])
        sral.store_interpretation(agent.name, result["retrieved_docs"], result["analysis"])
        iakb.publish(agent.name, ftype, result["analysis"][:200].replace("\n"," "), result["confidence"], notify)
        results[specialty] = result
    conflicts = scdp.detect_and_resolve(results)
    dcwo_report = dcwo.run_consensus(results, conflicts)

    afl = AFL(confidence_threshold=0.65, max_iterations=1)
    agent_map = {"cardiology": agents["cardiology"], "nephrology": agents["nephrology"],
                 "diabetology": agents["diabetology"], "pharmacology": agents["pharmacology"]}
    results = afl.run_feedback_loop(results, agent_map, dcwo_report, sral, iakb, scdp, dcwo, question_text)
    conflicts = scdp.detect_and_resolve(results)

    question_snippet = question_text.split("Abstract Context:")[0][:200]
    web_evidence = get_web_evidence(question_snippet)

    try:
        fused = synthesize_final_answer(question_text, results, conflicts, web_evidence)
        if fused != "unclear":
            return fused
    except Exception as e:
        print(f"  [Fusion] failed, falling back to confidence vote: {e}")

    votes = {}
    for r in results.values():
        ans = extract_answer(r["analysis"])
        if ans != "unclear":
            votes[ans] = votes.get(ans, 0) + r["confidence"]
    return max(votes, key=votes.get) if votes else "unclear"

def get_baseline_answer(question_text, agent):
    return extract_answer(analyze_with_retry(agent, question_text)["analysis"])

def run_benchmark(num_questions=50):
    print("\n"+"="*60)
    print("MARC-Clinical PubMedQA Benchmark")
    print(f"Evaluating {num_questions} PubMedQA questions | BioBERT")
    print("="*60)

    progress_path = "evaluation/pubmedqa_progress.json"
    completed = {}
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                saved = json.load(f)
            completed = {r["question_id"]: r for r in saved.get("detailed_results", [])}
            if completed:
                print(f"\n🔄 RESUMING: found {len(completed)} already-completed questions in "
                      f"{progress_path} — skipping those, continuing from where it left off.")
        except Exception as e:
            print(f"  (Could not read existing progress file, starting fresh: {e})")

    try:
        from datasets import load_dataset
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        questions = list(dataset)[:num_questions]
        question_source = "real_pubmedqa_huggingface"
        print(f"\n{'='*60}")
        print(f"✅ REAL PubMedQA loaded from HuggingFace: {len(questions)} questions")
        print(f"   This number IS comparable to the base paper's 88% PubMedQA figure")
        print(f"   (scored via exact yes/no/maybe match, not cosine similarity - see")
        print(f"   docstring at top of this file for why that's a valid deviation).")
        print(f"{'='*60}")
        sample = questions[0]
        print(f"   Sample record fields: {list(sample.keys())}")
    except Exception as e:
        question_source = "FAILED_TO_LOAD"
        print(f"\n{'!'*60}")
        print(f"❌ FATAL: Could not load real PubMedQA ({e})")
        print(f"❌ There is no in-domain fallback for this dataset (unlike MedQA) -")
        print(f"❌ fix the HuggingFace load (check `datasets` package, internet access,")
        print(f"❌ or try split='train' vs other splits) before this benchmark can run.")
        print(f"{'!'*60}")
        return None

    print("\nInitializing agents...")
    agents = {
        "cardiology":  CardiologyAgent(KB["cardiology"]),
        "diabetology": DiabetologyAgent(KB["diabetology"]),
        "nephrology":  NephrologyAgent(KB["nephrology"]),
        "pharmacology":PharmacologyAgent(KB["pharmacology"]),
    }
    print("All agents ready.\n")

    marc_correct = sum(1 for r in completed.values() if r["marc_correct"])
    baseline_correct = sum(1 for r in completed.values() if r["baseline_correct"])
    detailed = list(completed.values())

    for i, q in enumerate(questions):
        qid = i + 1
        if qid in completed:
            continue
        print(f"\n[{qid}/{num_questions}] {q['question'][:65]}...")
        correct = get_correct_answer(q)
        qtext   = format_question(q)

        sral = SRAL(); iakb = IAKB(); scdp = SCDP()
        dcwo = DCWO(convergence_threshold=0.75, max_rounds=3)
        for sub, types in [
            ("Cardiology Agent",   ["renal_function","drug_safety","fluid_status"]),
            ("Diabetology Agent",  ["renal_function","cardiac_function","drug_safety"]),
            ("Nephrology Agent",   ["cardiac_function","drug_safety","glucose_control"]),
            ("Pharmacology Agent", ["renal_function","cardiac_function","glucose_control","drug_safety"]),
        ]:
            iakb.subscribe(sub, types)

        marc_ans   = get_marc_answer(qtext, agents, sral, iakb, scdp, dcwo)
        marc_right = marc_ans == correct
        if marc_right: marc_correct += 1

        base_ans   = get_baseline_answer(qtext, agents["cardiology"])
        base_right = base_ans == correct
        if base_right: baseline_correct += 1

        detailed.append({"question_id":qid,"question":q["question"][:100],
                         "correct":correct,"marc_answer":marc_ans,"marc_correct":marc_right,
                         "baseline_answer":base_ans,"baseline_correct":base_right})

        rm = round(marc_correct/qid*100,1)
        rb = round(baseline_correct/qid*100,1)
        print(f"  Correct:{correct} | MARC:{marc_ans}{'✓' if marc_right else '✗'} "
              f"| Base:{base_ans}{'✓' if base_right else '✗'} "
              f"| Running MARC={rm}% Base={rb}%")

        os.makedirs("evaluation", exist_ok=True)
        with open(progress_path, "w") as f:
            json.dump({"detailed_results": detailed}, f, indent=2)

        time.sleep(2)

    marc_acc = round(marc_correct/num_questions*100,1)
    base_acc = round(baseline_correct/num_questions*100,1)
    gain     = round(marc_acc-base_acc,1)

    print("\n"+"="*60)
    print("FINAL BENCHMARK RESULTS - PubMedQA")
    print("="*60)
    print(f"MARC Multi-Agent Accuracy : {marc_acc}%  ({marc_correct}/{num_questions})")
    print(f"Single-Agent Baseline     : {base_acc}% ({baseline_correct}/{num_questions})")
    print(f"MARC Improvement          : +{gain}%")
    print(f"Base Paper Target(PubMedQA): 88.0%")
    print(f"Beat base paper?          : {'YES ✅' if marc_acc>=88 else f'Gap: {round(88-marc_acc,1)}%'}")
    print("="*60)

    output = {"benchmark":"PubMedQA","embeddings":"BioBERT",
              "question_source":question_source,
              "scoring_method":"exact_match_yes_no_maybe (NOT cosine similarity like base paper)",
              "comparable_to_base_paper":question_source=="real_pubmedqa_huggingface",
              "num_questions":num_questions,"marc_accuracy":marc_acc,
              "marc_correct":marc_correct,"baseline_accuracy":base_acc,
              "baseline_correct":baseline_correct,"improvement":gain,
              "base_paper_target":88.0,"beat_base_paper":marc_acc>=88,
              "detailed_results":detailed}
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/pubmedqa_results.json","w") as f:
        json.dump(output,f,indent=2)
    print("Results saved → evaluation/pubmedqa_results.json")
    return output

if __name__ == "__main__":
    run_benchmark(num_questions=50)
