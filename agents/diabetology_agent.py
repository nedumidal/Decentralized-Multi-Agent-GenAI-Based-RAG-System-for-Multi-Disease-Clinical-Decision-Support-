import os
import re
import requests
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
    """Live PubMed search — base paper's Evidence-Based Scanner technique.
    No API key needed — uses free NCBI E-utilities."""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        resp = requests.get(search_url, params={
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance",
            "datetype": "pdat", "mindate": "2022", "maxdate": "2025"
        }, timeout=8)
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return ""
        fetch_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids[:max_results]),
                    "rettype": "abstract", "retmode": "text"}, timeout=10)
        text = fetch_resp.text.strip()
        if len(text) > 1500:
            text = text[:1500] + "..."
        return f"\n[PubMed Evidence ({len(pmids)} articles)]:\n{text}\n"
    except Exception:
        return ""


class DiabetologyAgent:
    def __init__(self, pdf_path: str):
        self.name = "Diabetology Agent"
        self.specialty = "diabetology"
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

        # Switched to gemini-3.6-flash — gemini-3.7-flash is still rollout-capacity-
        # constrained (503 "high demand" + hard ~20-23 req/day ceiling observed in
        # Cloud Console dashboard despite active billing). 3.6-flash has been GA since
        # July 21, 2026 and has had time to reach standard Tier 1 throughput.
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=90
        )

        prompt = PromptTemplate.from_template("""You are a specialist Diabetologist AI agent.
Analyze this clinical case using ADA Standards of Medical Care in Diabetes 2025 context provided.
Extract ALL relevant treatment recommendations. Cite specific sections and page numbers.
Focus on: HbA1c targets, SGLT2 inhibitors, GLP-1 receptor agonists, metformin use,
insulin therapy, diabetes management in CKD and heart failure, glucose monitoring.
Even if context is partial, extract what is available.
For USMLE-style questions, identify the single best answer option (A/B/C/D) explicitly.

Context: {context}

Clinical Case: {question}

Diabetology Analysis (with citations):
End your response with a line in EXACTLY this format (no extra text on that line):
CONFIDENCE: 0.XX
Where 0.XX is your calibrated confidence (0.0-1.0) based on how well the retrieved context matches the case.
Use LOW confidence (0.1-0.4) if context is off-topic or insufficient.
Use HIGH confidence (0.8-0.95) only when guidelines directly support your answer.""")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def _get_pubmed_query(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        if "hba1c" in case_lower or "diabetes" in case_lower or "t2dm" in case_lower:
            return "type 2 diabetes HbA1c treatment guidelines 2024"
        if "sglt2" in case_lower or "empagliflozin" in case_lower or "dapagliflozin" in case_lower:
            return "SGLT2 inhibitor diabetes heart failure kidney outcomes"
        if "glp" in case_lower or "semaglutide" in case_lower or "liraglutide" in case_lower:
            return "GLP-1 receptor agonist diabetes cardiovascular outcomes"
        if "metformin" in case_lower:
            return "metformin CKD contraindication dose adjustment"
        if "insulin" in case_lower:
            return "insulin therapy diabetes initiation management"
        return "type 2 diabetes management guidelines 2024"

    def _get_focused_query(self) -> str:
        return "diabetes HbA1c treatment SGLT2 inhibitor GLP1 receptor agonist metformin CKD heart failure glucose management insulin"

    def _get_query_for_case(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        terms = []
        if "hba1c" in case_lower or "diabetes" in case_lower or "t2dm" in case_lower:
            terms.append("diabetes mellitus T2DM HbA1c target treatment guidelines")
        if "sglt2" in case_lower or "empagliflozin" in case_lower or "dapagliflozin" in case_lower:
            terms.append("SGLT2 inhibitor diabetes heart failure kidney benefit")
        if "glp" in case_lower or "semaglutide" in case_lower or "liraglutide" in case_lower:
            terms.append("GLP-1 receptor agonist cardiovascular outcomes diabetes")
        if "metformin" in case_lower:
            terms.append("metformin CKD contraindication dose adjustment eGFR")
        if "insulin" in case_lower:
            terms.append("insulin therapy diabetes management initiation")
        if "ckd" in case_lower or "kidney" in case_lower or "egfr" in case_lower:
            terms.append("diabetes kidney disease CKD management glucose control")
        if terms:
            return " ".join(terms)
        return self._get_focused_query()

    def analyze(self, clinical_case: str) -> dict:
        print(f"[{self.name}] Analyzing case...")
        retrieval_query = self._get_query_for_case(clinical_case)
        retrieved_docs = self.retriever.invoke(retrieval_query)
        web_evidence = fetch_pubmed_evidence(self._get_pubmed_query(clinical_case))
        if web_evidence:
            print(f"[{self.name}] PubMed evidence fetched ✓")
        augmented_case = clinical_case + web_evidence
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
