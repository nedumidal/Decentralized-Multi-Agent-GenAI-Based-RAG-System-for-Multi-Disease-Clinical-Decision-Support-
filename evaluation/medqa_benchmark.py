"""
MedQA Benchmark Evaluation for MARC-Clinical
"""
import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
# override=True: the .env file wins over any key already set in the Windows environment.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True, encoding="utf-8-sig")  # utf-8-sig handles a hidden BOM at the top of .env

def _redact(s):
    """Hide API keys that Google/Groq echo back inside error messages."""
    s = re.sub(r"api_key:[^\s'\"\\]+", "api_key:<redacted>", s)
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "<redacted>", s)
    s = re.sub(r"AQ\.[0-9A-Za-z_\-]{20,}", "<redacted>", s)
    s = re.sub(r"gsk_[0-9A-Za-z]{20,}", "<redacted>", s)
    return s

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

def get_sample_questions(n=50):
    samples = [
        {"question":"A 58-year-old male with HFrEF (EF 35%) and BNP 890. Which drug reduces mortality?","options":{"A":"Amlodipine","B":"Carvedilol","C":"Digoxin","D":"Hydralazine"},"answer_idx":"B"},
        {"question":"Patient with CKD eGFR 42 and T2DM HbA1c 8.2%. Best agent for cardio and renoprotection?","options":{"A":"Glipizide","B":"Sitagliptin","C":"Empagliflozin","D":"Pioglitazone"},"answer_idx":"C"},
        {"question":"HFrEF patient on ACE inhibitor develops cough. Best substitute?","options":{"A":"Beta blocker","B":"ARB","C":"CCB","D":"Aldosterone antagonist"},"answer_idx":"B"},
        {"question":"First-line diuretic for fluid overload in heart failure?","options":{"A":"HCTZ","B":"Spironolactone","C":"Furosemide","D":"Acetazolamide"},"answer_idx":"C"},
        {"question":"HFrEF EF 30% on optimal GDMT with NYHA III symptoms. Device therapy?","options":{"A":"LVAD","B":"ICD","C":"Pacemaker","D":"Transplant"},"answer_idx":"B"},
        {"question":"Which SGLT2 inhibitor approved for HFrEF regardless of diabetes?","options":{"A":"Canagliflozin","B":"Empagliflozin","C":"Ertugliflozin","D":"Ipragliflozin"},"answer_idx":"B"},
        {"question":"CKD eGFR 42 with hypertension and proteinuria. First-line antihypertensive?","options":{"A":"Amlodipine","B":"Metoprolol","C":"Lisinopril","D":"Furosemide"},"answer_idx":"C"},
        {"question":"SGLT2 inhibitor in CKD. At what eGFR should it NOT be initiated?","options":{"A":"<45","B":"<30","C":"<20","D":"<15"},"answer_idx":"C"},
        {"question":"HFrEF on beta blocker and ACEi. Drug to reduce hospitalization?","options":{"A":"Digoxin","B":"Spironolactone","C":"Hydralazine","D":"Nifedipine"},"answer_idx":"B"},
        {"question":"Metformin should be avoided in CKD when eGFR falls below?","options":{"A":"<60","B":"<45","C":"<30","D":"<20"},"answer_idx":"C"},
        {"question":"Best GLP-1 RA for cardiovascular risk reduction in T2DM?","options":{"A":"Exenatide","B":"Dulaglutide","C":"Semaglutide","D":"Liraglutide"},"answer_idx":"C"},
        {"question":"MRA therapy in HFrEF contraindicated when eGFR is?","options":{"A":"<60","B":"<45","C":"<30","D":"<20"},"answer_idx":"C"},
        {"question":"Which beta blocker is preferred in HFrEF?","options":{"A":"Atenolol","B":"Metoprolol tartrate","C":"Carvedilol","D":"Propranolol"},"answer_idx":"C"},
        {"question":"Target systolic BP in CKD per KDIGO 2024?","options":{"A":"<140","B":"<130","C":"<120","D":"<110"},"answer_idx":"C"},
        {"question":"Drug blocking angiotensin and neprilysin in HFrEF?","options":{"A":"Valsartan","B":"Sacubitril/Valsartan","C":"Losartan","D":"Olmesartan"},"answer_idx":"B"},
        {"question":"HbA1c target for T2DM with HFrEF and CKD?","options":{"A":"<6.0%","B":"<6.5%","C":"<7.0%","D":"<8.0%"},"answer_idx":"C"},
        {"question":"ACE inhibitor needing dose reduction in CKD due to renal clearance?","options":{"A":"Fosinopril","B":"Trandolapril","C":"Enalapril","D":"Benazepril"},"answer_idx":"C"},
        {"question":"Electrolyte monitored when starting MRA in HFrEF?","options":{"A":"Sodium","B":"Calcium","C":"Potassium","D":"Magnesium"},"answer_idx":"C"},
        {"question":"Finerenone non-steroidal MRA indicated for?","options":{"A":"HFpEF only","B":"CKD with T2DM","C":"Hypertension alone","D":"HFrEF only"},"answer_idx":"B"},
        {"question":"Loop diuretic efficacy reduced in CKD because of?","options":{"A":"Reduced protein binding","B":"Impaired tubular secretion","C":"Increased hepatic metabolism","D":"Enhanced renal excretion"},"answer_idx":"B"},
        {"question":"Drug combination contraindicated due to hyperkalemia in CKD?","options":{"A":"ACEi + beta blocker","B":"ACEi + ARB","C":"Beta blocker + CCB","D":"Diuretic + ACEi"},"answer_idx":"B"},
        {"question":"CKD staging G3b corresponds to eGFR of?","options":{"A":"60-89","B":"45-59","C":"30-44","D":"15-29"},"answer_idx":"C"},
        {"question":"First-line for microalbuminuria in diabetic nephropathy?","options":{"A":"CCB","B":"ACEi or ARB","C":"Beta blocker","D":"Loop diuretic"},"answer_idx":"B"},
        {"question":"Drug reducing CKD progression in T2DM with eGFR 42?","options":{"A":"Sitagliptin","B":"Glipizide","C":"Dapagliflozin","D":"Insulin"},"answer_idx":"C"},
        {"question":"NSAIDs contraindicated in HFrEF because they?","options":{"A":"Cause bradycardia","B":"Cause sodium retention worsening HF","C":"Increase diuresis","D":"Block beta receptors"},"answer_idx":"B"},
        {"question":"Beta-blockers mask hypoglycemia symptoms except?","options":{"A":"Tachycardia","B":"Palpitations","C":"Sweating","D":"Tremor"},"answer_idx":"C"},
        {"question":"Bisoprolol initial dose in HFrEF?","options":{"A":"5mg daily","B":"2.5mg daily","C":"1.25mg daily","D":"10mg daily"},"answer_idx":"C"},
        {"question":"Antidiabetic with highest hypoglycemia risk?","options":{"A":"SGLT2 inhibitor","B":"GLP-1 RA","C":"Sulfonylurea","D":"Metformin"},"answer_idx":"C"},
        {"question":"Tirzepatide agonizes which receptors?","options":{"A":"GLP-1 only","B":"GIP and GLP-1","C":"GIP only","D":"DPP-4 and GLP-1"},"answer_idx":"B"},
        {"question":"HF medication causing angioedema?","options":{"A":"Beta blocker","B":"MRA","C":"ACE inhibitor","D":"ARNI"},"answer_idx":"C"},
        {"question":"Ivabradine used in HFrEF when HR remains above?","options":{"A":"60 bpm","B":"70 bpm","C":"80 bpm","D":"90 bpm"},"answer_idx":"B"},
        {"question":"CKD complication from reduced EPO production?","options":{"A":"Hyperkalemia","B":"Anemia","C":"Metabolic acidosis","D":"Hypertension"},"answer_idx":"B"},
        {"question":"Hyperkalemia management in CKD on ACEi and MRA?","options":{"A":"Increase diuretic","B":"Add potassium supplement","C":"Reduce or stop MRA","D":"Add calcium gluconate"},"answer_idx":"C"},
        {"question":"Drug reducing CV death in HFrEF EF below 35%?","options":{"A":"Digoxin","B":"Amiodarone","C":"Eplerenone","D":"Nifedipine"},"answer_idx":"C"},
        {"question":"Oral GLP-1 RA formulation?","options":{"A":"Liraglutide","B":"Oral Semaglutide","C":"Dulaglutide","D":"Exenatide"},"answer_idx":"B"},
        {"question":"ACEi+ARB combination not recommended because?","options":{"A":"No benefit","B":"Increased hypotension and renal failure","C":"Cost issues","D":"Causes bradycardia"},"answer_idx":"B"},
        {"question":"Best marker for cardiorenal syndrome severity?","options":{"A":"Troponin","B":"BNP alone","C":"Creatinine rise during HF treatment","D":"BNP and creatinine together"},"answer_idx":"D"},
        {"question":"Empagliflozin in HFrEF reduces which outcome?","options":{"A":"Stroke","B":"All-cause mortality","C":"HF hospitalization and CV death","D":"Arrhythmia"},"answer_idx":"C"},
        {"question":"Investigation confirming CKD staging accurately?","options":{"A":"Serum creatinine alone","B":"eGFR + urine ACR","C":"BUN only","D":"Cystatin C alone"},"answer_idx":"B"},
        {"question":"Vasodilator for HFrEF when ACEi/ARB not tolerated?","options":{"A":"Amlodipine + metoprolol","B":"Hydralazine + isosorbide dinitrate","C":"Nifedipine + propranolol","D":"Diltiazem + furosemide"},"answer_idx":"B"},
        {"question":"Drug held before iodinated contrast in CKD?","options":{"A":"Amlodipine","B":"Metformin","C":"Metoprolol","D":"Furosemide"},"answer_idx":"B"},
        {"question":"Potassium level requiring MRA discontinuation in HFrEF?","options":{"A":">4.5","B":">5.0","C":">5.5","D":">6.0"},"answer_idx":"B"},
        {"question":"HbA1c target for elderly T2DM with multiple comorbidities?","options":{"A":"<6.5%","B":"<7.0%","C":"<7.5-8.0%","D":"<9.0%"},"answer_idx":"C"},
        {"question":"ACR >300 mg/g indicates?","options":{"A":"Normal","B":"Microalbuminuria","C":"Macroalbuminuria","D":"Nephrotic range"},"answer_idx":"C"},
        {"question":"Preferred antihypertensive drug class in HFrEF?","options":{"A":"CCB","B":"Alpha blocker","C":"ACEi/ARB","D":"Central acting"},"answer_idx":"C"},
        {"question":"Maximum carvedilol dose in HFrEF?","options":{"A":"12.5mg BD","B":"25mg BD","C":"50mg BD","D":"6.25mg BD"},"answer_idx":"B"},
        {"question":"Sodium restriction target in stable HFrEF?","options":{"A":"<1g/day","B":"<2g/day","C":"<3g/day","D":"No restriction"},"answer_idx":"B"},
        {"question":"SGLT2 inhibitor renal protection main mechanism?","options":{"A":"ACE inhibition","B":"Tubuloglomerular feedback restoration","C":"ARB effect","D":"Diuresis only"},"answer_idx":"B"},
        {"question":"Electrolyte imbalance worsening arrhythmia risk in HF?","options":{"A":"Hypernatremia","B":"Hypokalemia","C":"Hypercalcemia","D":"Hypermagnesemia"},"answer_idx":"B"},
        {"question":"First monitoring after starting ACEi in CKD?","options":{"A":"LFTs at 1 month","B":"Serum K and creatinine at 1-2 weeks","C":"CBC at 2 weeks","D":"Urine culture at 1 week"},"answer_idx":"B"},
    ]
    result = []
    for i in range(n):
        result.append(samples[i % len(samples)])
    return result

def format_question(q):
    options = q.get("options", {})
    if isinstance(options, dict):
        opt_str = "\n".join([f"  {k}) {v}" for k, v in options.items()])
    else:
        opt_str = str(options)
    return (f"Medical Question (USMLE):\n{q['question']}\n\nOptions:\n{opt_str}\n\n"
            "Select best answer. Start with answer letter (A/B/C/D) only.")

def extract_answer(text):
    """Robust letter extraction. Returns 'X' when no clear answer letter is found.
    Never guesses from stray letters inside words (the old fallback scanned the first
    80 characters and could silently score 'A' from words like 'ANSWER')."""
    if not text:
        return "X"
    t = text.strip()
    if not t or t.lower().startswith("unavailable"):  # agent's failure marker only, not the word inside a real analysis
        return "X"
    # 1. Answer on the first non-empty line: "B", "B)", "**B**", "Answer: B"
    first = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"^[\s\*\#\(\[]*(?:(?i:(?:best |correct |final )?answer)\s*(?i:is)?\s*[:\-]?\s*)?[\*\(\[]*([ABCD])(?=\s*(?:[\)\.\:\*\]]|-|$))", first)
    if m:
        return m.group(1)
    # 2. Explicit marker anywhere: "Answer: B", "the answer is (B)"
    m = re.search(r"(?i:(?:best |correct |final )?answer\s*(?:is)?)\s*[:\-]?\s*[\*\(\[]*\s*([ABCD])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    # 3. A single distinct parenthesised letter, e.g. "(D)"
    letters = set(re.findall(r"\(([ABCD])\)", t))
    if len(letters) == 1:
        return letters.pop()
    return "X"

def get_correct_answer(q):
    for field in ["answer_idx","answer","correct_answer","label"]:
        if field in q:
            val = str(q[field]).strip().upper()
            if val in ["A","B","C","D"]:
                return val
            if val in ["0","1","2","3"]:
                return ["A","B","C","D"][int(val)]
    raise ValueError(f"No valid answer key found in question record: {list(q.keys())}")

def analyze_with_retry(agent, case, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return agent.analyze(case)
        except Exception as e:
            last_error = e
            es = str(e)
            print(f"  [{agent.name}] error: {_redact(es)[:300]}")
            if "PERMISSION_DENIED" in es or "403" in es:
                print(f"  [{agent.name}] account/billing-level block — retrying won't help. "
                      f"Check your Google Cloud project's billing status. Skipping retries.")
                break
            wait = 20 if "503" in es or "UNAVAILABLE" in es else (40 if "429" in es else 10)
            if attempt == max_retries-1:
                break
            print(f"  Retry {attempt+1} in {wait}s...")
            time.sleep(wait)
    print(f"  [{agent.name}] giving up. Last error: {_redact(str(last_error))[:300]}")
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
same USMLE-style question below. Your job is to weigh their reasoning, resolve any disagreement between
them, and output the single best final answer.

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
- Not every specialist's domain is relevant to every question. If a question is not about that agent's
  specialty, down-weight their reasoning even if their stated confidence is high — expertise relevance
  matters more than raw confidence.
- Treat the Live Web Evidence as an independent cross-check against the specialist agents, the same way
  a clinician would sanity-check a guideline against a recent source. If web evidence is empty or clearly
  irrelevant, rely on the specialist agents alone.
- If the agents agree, confirm the consensus letter.
- If they disagree, use independent clinical reasoning grounded in the question stem to decide, don't just
  pick the majority.
- Respond with ONLY the final answer letter (A/B/C/D) on the first line, followed by a one-sentence
  justification on the next line.""")

def synthesize_final_answer(question_text, results, conflicts, web_evidence=""):
    """LLM-based fusion step — mirrors the base paper's Clinical Data Fusion Agent,
    which their own ablation study showed was the single highest-impact component
    (removing it dropped their accuracy from 94.7% to 60%). Also mirrors their
    Evidence-Based Scanner Agent by cross-checking against live web evidence —
    their own numbers showed web evidence alone (70.0%) beat static RAG alone
    (65.0%) on MedQA."""
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
        time.sleep(3)  # pacing — avoids bursting into per-minute rate limits right after a billing upgrade
    conflicts = scdp.detect_and_resolve(results)
    dcwo_report = dcwo.run_consensus(results, conflicts)

    # AFL: re-query low-confidence agents (anchored to this question's specific
    # values) before locking in the final answer. Capped at 1 iteration for the
    # benchmark (vs 2-3 in the main.py demo) to keep total API calls manageable
    # across a 50-question run — round-2 gains were marginal in testing anyway.
    afl = AFL(confidence_threshold=0.65, max_iterations=1)
    agent_map = {"cardiology": agents["cardiology"], "nephrology": agents["nephrology"],
                 "diabetology": agents["diabetology"], "pharmacology": agents["pharmacology"]}
    results = afl.run_feedback_loop(results, agent_map, dcwo_report, sral, iakb, scdp, dcwo, question_text)
    conflicts = scdp.detect_and_resolve(results)

    # Live web evidence, mirroring the base paper's Evidence-Based Scanner Agent.
    # One search per question (not per-agent) — keeps this cheap and fast since
    # ddgs is free/keyless, and the fusion step is where cross-checking belongs.
    question_snippet = question_text.split("Options:")[0][:200]
    web_evidence = get_web_evidence(question_snippet)
    time.sleep(2)  # pacing before the fusion call

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
    time.sleep(2)  # pacing — this fires right after get_marc_answer's own calls
    return extract_answer(analyze_with_retry(agent, question_text)["analysis"])

def run_benchmark(num_questions=50):
    print("\n"+"="*60)
    print("MARC-Clinical MedQA Benchmark")
    print(f"Evaluating {num_questions} USMLE questions | BioBERT")
    print("="*60)

    progress_path = "evaluation/medqa_progress.json"
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
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        questions = list(dataset)[:num_questions]
        question_source = "real_medqa_huggingface"
        print(f"\n{'='*60}")
        print(f"✅ REAL MedQA-USMLE loaded from HuggingFace: {len(questions)} questions")
        print(f"   This number IS comparable to the base paper's 94% MedQA figure.")
        print(f"{'='*60}")
        # Sanity check field names match what get_correct_answer()/format_question() expect
        sample = questions[0]
        print(f"   Sample record fields: {list(sample.keys())}")
    except Exception as e:
        question_source = "builtin_fallback_samples"
        print(f"\n{'!'*60}")
        print(f"⚠️  WARNING: Could not load real MedQA-USMLE ({e})")
        print(f"⚠️  Falling back to {num_questions} BUILT-IN cardio/renal/diabetes/pharm")
        print(f"⚠️  questions. These are IN-DOMAIN for your 4 agents and are")
        print(f"⚠️  NOT a valid, publishable comparison against the base paper's")
        print(f"⚠️  94% MedQA figure. Fix the HuggingFace load before using this")
        print(f"⚠️  run's number in any report or paper.")
        print(f"{'!'*60}")
        questions = get_sample_questions(num_questions)

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

        # Checkpoint after EVERY question — a hang, crash, or Ctrl+C from here loses
        # at most the one in-flight question, never the whole run.
        os.makedirs("evaluation", exist_ok=True)
        with open(progress_path, "w") as f:
            json.dump({"detailed_results": detailed}, f, indent=2)

        time.sleep(2)

    marc_acc = round(marc_correct/num_questions*100,1)
    base_acc = round(baseline_correct/num_questions*100,1)
    gain     = round(marc_acc-base_acc,1)

    print("\n"+"="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"MARC Multi-Agent Accuracy : {marc_acc}%  ({marc_correct}/{num_questions})")
    print(f"Single-Agent Baseline     : {base_acc}% ({baseline_correct}/{num_questions})")
    print(f"MARC Improvement          : +{gain}%")
    print(f"Base Paper Target(MedQA)  : 94.0%")
    print(f"Beat base paper?          : {'YES ✅' if marc_acc>=94 else f'Gap: {round(94-marc_acc,1)}%'}")
    print("="*60)

    output = {"benchmark":"MedQA-USMLE","embeddings":"BioBERT",
              "question_source":question_source,
              "comparable_to_base_paper":question_source=="real_medqa_huggingface",
              "num_questions":num_questions,"marc_accuracy":marc_acc,
              "marc_correct":marc_correct,"baseline_accuracy":base_acc,
              "baseline_correct":baseline_correct,"improvement":gain,
              "base_paper_target":94.0,"beat_base_paper":marc_acc>=94,
              "detailed_results":detailed}
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/medqa_results.json","w") as f:
        json.dump(output,f,indent=2)
    print("Results saved → evaluation/medqa_results.json")
    return output

if __name__ == "__main__":
    run_benchmark(num_questions=50)
