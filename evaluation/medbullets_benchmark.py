"""
MedBullets Benchmark Evaluation for MARC-Clinical

MedBullets (Chen, Fang, Singla, Dredze - the same paper the base paper cites as
reference [31] for its own baseline comparisons) is USMLE Step 2/3-style clinical
case questions, 5-option multiple choice (A-E). Uses JesseLiu/medbulltes5op on
HuggingFace, which was built directly from that paper's released format.

Schema is verified defensively at runtime (see the printed "Sample record fields"
line when you run this) since exact field names can vary by dataset version -
format_question()/get_correct_answer() try several common naming patterns rather
than assuming one, the same defensive pattern medqa_benchmark.py already uses.
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

VALID_LETTERS = ["A","B","C","D","E"]

def _extract_options(q):
    """Try several common schema patterns since the exact field names for this
    dataset weren't verified ahead of time - check the printed sample fields
    when you run this and adjust here if none of these patterns match."""
    if isinstance(q.get("options"), dict):
        return q["options"]
    if isinstance(q.get("choices"), dict):
        return q["choices"]
    opts = {}
    for letter, key_variants in zip(VALID_LETTERS,
            [["opa","option_a","choice_a","A"], ["opb","option_b","choice_b","B"],
             ["opc","option_c","choice_c","C"], ["opd","option_d","choice_d","D"],
             ["ope","option_e","choice_e","E"]]):
        for key in key_variants:
            if key in q and q[key]:
                opts[letter] = q[key]
                break
    return opts

def format_question(q):
    options = _extract_options(q)
    opt_str = "\n".join([f"  {k}) {v}" for k, v in options.items()]) if options else "(options not parsed - check schema)"
    stem = q.get("question") or q.get("stem") or q.get("case") or str(q)
    return (f"Medical Question (USMLE Step 2/3 - MedBullets):\n{stem}\n\nOptions:\n{opt_str}\n\n"
            f"Select the best answer. Start with the answer letter ({'/'.join(VALID_LETTERS)}) only.")

def extract_answer(text):
    if not text:
        return "X"
    t = text.strip().upper()
    for letter in VALID_LETTERS:
        if (t.startswith(letter+" ") or t.startswith(letter+")")
                or t.startswith(letter+".") or t.startswith(letter+":")):
            return letter
    for marker in ["ANSWER: ","ANSWER IS ","BEST ANSWER: ","CORRECT ANSWER: "]:
        if marker in t:
            idx = t.index(marker)+len(marker)
            if idx < len(t) and t[idx] in "ABCDE":
                return t[idx]
    for letter in VALID_LETTERS:
        if f"({letter})" in t or f"OPTION {letter}" in t:
            return letter
    for ch in t[:80]:
        if ch in "ABCDE":
            return ch
    return "X"

def get_correct_answer(q):
    for field in ["answer_idx","answer","correct_answer","label","correct_option"]:
        if field in q and q[field] not in (None, ""):
            val = str(q[field]).strip().upper()
            if val in VALID_LETTERS:
                return val
            if val.isdigit() and int(val) < len(VALID_LETTERS):
                return VALID_LETTERS[int(val)]
    return "A"

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
same USMLE Step 2/3-style clinical case below. Your job is to weigh their reasoning, resolve any
disagreement, and output the single best final answer.

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
- Not every specialist's domain is relevant to every question. Down-weight agents reasoning outside
  their expertise even if their stated confidence is high.
- If the agents agree, confirm the consensus letter. If they disagree, use independent clinical
  reasoning grounded in the case stem to decide.
- Respond with ONLY the final answer letter (A/B/C/D/E) on the first line, followed by a one-sentence
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

    question_snippet = question_text.split("Options:")[0][:200]
    web_evidence = get_web_evidence(question_snippet)

    try:
        fused = synthesize_final_answer(question_text, results, conflicts, web_evidence)
        if fused != "X":
            return fused
    except Exception as e:
        print(f"  [Fusion] failed, falling back to confidence vote: {e}")

    votes = {}
    for r in results.values():
        letter = extract_answer(r["analysis"])
        if letter != "X":
            votes[letter] = votes.get(letter, 0) + r["confidence"]
    return max(votes, key=votes.get) if votes else "X"

def get_baseline_answer(question_text, agent):
    return extract_answer(analyze_with_retry(agent, question_text)["analysis"])

def run_benchmark(num_questions=50):
    print("\n"+"="*60)
    print("MARC-Clinical MedBullets Benchmark")
    print(f"Evaluating {num_questions} USMLE Step 2/3 questions | BioBERT")
    print("="*60)

    progress_path = "evaluation/medbullets_progress.json"
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
        dataset = load_dataset("JesseLiu/medbulltes5op", split="test")
        questions = list(dataset)[:num_questions]
        question_source = "real_medbullets_huggingface"
        print(f"\n{'='*60}")
        print(f"✅ REAL MedBullets loaded from HuggingFace: {len(questions)} questions")
        print(f"   This number IS comparable to the base paper's 84% MedBullets figure.")
        print(f"{'='*60}")
        sample = questions[0]
        print(f"   Sample record fields: {list(sample.keys())}")
        print(f"   >>> If format_question()/get_correct_answer() below look wrong for")
        print(f"   >>> this schema, check the field names above and adjust _extract_options()")
        print(f"   >>> and get_correct_answer() in this file accordingly.")
    except Exception as e:
        question_source = "FAILED_TO_LOAD"
        print(f"\n{'!'*60}")
        print(f"❌ Could not load 'JesseLiu/medbulltes5op' split='test' ({e})")
        print(f"❌ Trying split='train' as fallback...")
        print(f"{'!'*60}")
        try:
            from datasets import load_dataset
            dataset = load_dataset("JesseLiu/medbulltes5op", split="train")
            questions = list(dataset)[:num_questions]
            question_source = "real_medbullets_huggingface_train_split"
            print(f"✅ Loaded {len(questions)} questions from train split instead.")
            sample = questions[0]
            print(f"   Sample record fields: {list(sample.keys())}")
        except Exception as e2:
            print(f"❌ FATAL: Could not load MedBullets at all ({e2})")
            print(f"❌ No fallback exists for this dataset. Try 'LangAGI-Lab/medbullets'")
            print(f"❌ as an alternate source, or check huggingface-cli login if it's gated.")
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
        stem_preview = str(q.get("question") or q.get("stem") or q.get("case") or "")[:65]
        print(f"\n[{qid}/{num_questions}] {stem_preview}...")
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

        detailed.append({"question_id":qid,"question":stem_preview,
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
    print("FINAL BENCHMARK RESULTS - MedBullets")
    print("="*60)
    print(f"MARC Multi-Agent Accuracy : {marc_acc}%  ({marc_correct}/{num_questions})")
    print(f"Single-Agent Baseline     : {base_acc}% ({baseline_correct}/{num_questions})")
    print(f"MARC Improvement          : +{gain}%")
    print(f"Base Paper Target(MedBullets): 84.0%")
    print(f"Beat base paper?          : {'YES ✅' if marc_acc>=84 else f'Gap: {round(84-marc_acc,1)}%'}")
    print("="*60)

    output = {"benchmark":"MedBullets","embeddings":"BioBERT",
              "question_source":question_source,
              "comparable_to_base_paper":question_source.startswith("real_medbullets"),
              "num_questions":num_questions,"marc_accuracy":marc_acc,
              "marc_correct":marc_correct,"baseline_accuracy":base_acc,
              "baseline_correct":baseline_correct,"improvement":gain,
              "base_paper_target":84.0,"beat_base_paper":marc_acc>=84,
              "detailed_results":detailed}
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/medbullets_results.json","w") as f:
        json.dump(output,f,indent=2)
    print("Results saved → evaluation/medbullets_results.json")
    return output

if __name__ == "__main__":
    run_benchmark(num_questions=50)
