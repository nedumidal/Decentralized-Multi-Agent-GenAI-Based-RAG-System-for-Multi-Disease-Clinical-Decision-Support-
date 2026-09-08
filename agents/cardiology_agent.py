import os
import re
import requests
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading BioBERT embedding model (one time only)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("BioBERT embedding model loaded.")
    return _embeddings


def fetch_pubmed_evidence(query: str, max_results: int = 3) -> str:
    """
    Live PubMed search — base paper's Evidence-Based Scanner technique.
    No API key needed — uses free NCBI E-utilities.
    Fetches recent abstracts to supplement static RAG knowledge.
    """
    try:
        # Step 1: Search PubMed for relevant PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json",
            "sort": "relevance", "datetype": "pdat",
            "mindate": "2022", "maxdate": "2025"
        }
        resp = requests.get(search_url, params=search_params, timeout=8)
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return ""

        # Step 2: Fetch abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed", "id": ",".join(pmids[:max_results]),
            "rettype": "abstract", "retmode": "text"
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        abstract_text = fetch_resp.text.strip()

        # Truncate to avoid context overflow
        if len(abstract_text) > 1500:
            abstract_text = abstract_text[:1500] + "..."

        return f"\n[PubMed Evidence ({len(pmids)} articles)]:\n{abstract_text}\n"

    except Exception:
        return ""  # Silently fail — static RAG still works


class CardiologyAgent:
    def __init__(self, pdf_path: str):
        self.name = "Cardiology Agent"
        self.specialty = "cardiology"
        self.retriever = None
        self.chain = None
        self._build_knowledge_base(pdf_path)

    def _build_knowledge_base(self, pdf_path: str):
        print(f"[{self.name}] Loading knowledge base...")
        faiss_path = f"vectorstores_biobert/{self.specialty}"
        embeddings = get_embeddings()

        if os.path.exists(faiss_path):
            print(f"[{self.name}] Loading saved BioBERT vectorstore from disk...")
            vectorstore = FAISS.load_local(faiss_path, embeddings,
                                           allow_dangerous_deserialization=True)
            print(f"[{self.name}] Vectorstore loaded instantly.")
        else:
            print(f"[{self.name}] Building BioBERT vectorstore from PDF...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
            chunks = splitter.split_documents(documents)
            print(f"[{self.name}] Split into {len(chunks)} chunks.")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(faiss_path, exist_ok=True)
            vectorstore.save_local(faiss_path)
            print(f"[{self.name}] BioBERT vectorstore saved to disk.")

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        print(f"[{self.name}] Knowledge base ready.")

        # gemini-3.6-flash — established model (since Jul 21), stable Tier-1 capacity
        # for paying accounts, unlike gemini-3.7-flash which is still rollout-throttled
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=90
        )

        self.prompt = PromptTemplate.from_template("""You are a specialist Cardiologist AI agent.
Analyze this clinical case using the AHA/ACC Heart Failure Guidelines (RAG context)
AND the latest PubMed evidence provided below.
Cross-validate both sources. If they agree, state high confidence.
If they conflict, prefer the more recent PubMed evidence.
For USMLE-style questions, state the answer letter (A/B/C/D) in your FIRST line.
Cite sections and page numbers from guidelines.

RAG Context (Guidelines): {context}

Clinical Case: {question}

Cardiology Analysis:
End your response with EXACTLY this line:
CONFIDENCE: 0.XX""")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt | self.llm | StrOutputParser()
        )

    def _get_pubmed_query(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        if "heart failure" in case_lower or "hfref" in case_lower:
            return "heart failure HFrEF treatment guidelines 2024"
        if "beta blocker" in case_lower or "carvedilol" in case_lower:
            return "beta blocker heart failure mortality benefit"
        if "icd" in case_lower or "device therapy" in case_lower:
            return "ICD implantable cardioverter defibrillator heart failure EF 35"
        if "arni" in case_lower or "sacubitril" in case_lower:
            return "sacubitril valsartan ARNI heart failure outcomes"
        if "diuretic" in case_lower or "furosemide" in case_lower:
            return "loop diuretic decongestion heart failure management"
        return "heart failure treatment GDMT guidelines 2024"

    def _get_rag_query(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        terms = []
        if "heart failure" in case_lower or "hfref" in case_lower or "ef " in case_lower:
            terms.append("heart failure HFrEF treatment guidelines")
        if "beta blocker" in case_lower or "carvedilol" in case_lower:
            terms.append("beta blocker heart failure mortality")
        if "ace" in case_lower or "acei" in case_lower:
            terms.append("ACE inhibitor heart failure treatment")
        if "diuretic" in case_lower or "furosemide" in case_lower:
            terms.append("loop diuretic heart failure fluid")
        if "icd" in case_lower or "device" in case_lower:
            terms.append("ICD device therapy HFrEF EF 35")
        if "arni" in case_lower or "sacubitril" in case_lower:
            terms.append("sacubitril valsartan ARNI heart failure")
        if terms:
            return " ".join(terms)
        return "heart failure treatment HFrEF ejection fraction reduced GDMT"

    def analyze(self, clinical_case: str) -> dict:
        print(f"[{self.name}] Analyzing case...")

        # Step 1: RAG retrieval with case-specific query
        rag_query = self._get_rag_query(clinical_case)
        retrieved_docs = self.retriever.invoke(rag_query)

        # Step 2: Live PubMed evidence (base paper technique)
        pubmed_query = self._get_pubmed_query(clinical_case)
        web_evidence = fetch_pubmed_evidence(pubmed_query)
        if web_evidence:
            print(f"[{self.name}] PubMed evidence fetched ✓")

        # Step 3: Fuse RAG + web evidence into the question
        augmented_case = clinical_case + web_evidence

        # Step 4: LLM analysis
        answer = self.chain.invoke(augmented_case)

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "analysis": answer,
            "web_evidence_used": bool(web_evidence),
            "retrieved_docs": [
                {"doc_id": doc.metadata.get("page", "unknown"),
                 "source": doc.metadata.get("source", "unknown"),
                 "content_preview": doc.page_content[:150]}
                for doc in retrieved_docs
            ],
            "confidence": self._estimate_confidence(answer)
        }

    def _estimate_confidence(self, answer: str) -> float:
        match = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", answer, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        if "Not found" in answer and len(answer) < 100:
            return 0.2
        elif "recommend" in answer.lower() or "guideline" in answer.lower():
            return 0.85
        return 0.65
