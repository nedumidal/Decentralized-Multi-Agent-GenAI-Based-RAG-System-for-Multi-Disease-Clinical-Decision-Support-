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


class NephrologyAgent:
    def __init__(self, pdf_path: str):
        self.name = "Nephrology Agent"
        self.specialty = "nephrology"
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

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            timeout=90
        )

        self.prompt = PromptTemplate.from_template("""You are a specialist Nephrologist AI agent.
Analyze this clinical case using KDIGO CKD Guidelines 2024 (RAG context)
AND the latest PubMed evidence provided below.
Cross-validate both sources. Prefer recent PubMed evidence when guidelines conflict.
For USMLE-style questions, state the answer letter (A/B/C/D) in your FIRST line.
Cite sections and page numbers from guidelines.

RAG Context (Guidelines): {context}

Clinical Case: {question}

Nephrology Analysis:
End your response with EXACTLY this line:
CONFIDENCE: 0.XX""")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt | self.llm | StrOutputParser()
        )

    def _get_pubmed_query(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        if "egfr" in case_lower or "ckd" in case_lower or "kidney" in case_lower:
            return "chronic kidney disease CKD eGFR management treatment 2024"
        if "proteinuria" in case_lower or "albuminuria" in case_lower:
            return "proteinuria CKD treatment ACE inhibitor ARB guidelines"
        if "sglt2" in case_lower:
            return "SGLT2 inhibitor CKD progression prevention trial"
        if "blood pressure" in case_lower or "hypertension" in case_lower:
            return "hypertension CKD blood pressure target treatment"
        return "chronic kidney disease management guidelines 2024"

    def _get_rag_query(self, clinical_case: str) -> str:
        case_lower = clinical_case.lower()
        terms = []
        if "ckd" in case_lower or "kidney" in case_lower or "egfr" in case_lower:
            terms.append("CKD chronic kidney disease eGFR staging management")
        if "proteinuria" in case_lower or "albuminuria" in case_lower:
            terms.append("proteinuria ACR CKD treatment")
        if "ace" in case_lower or "arb" in case_lower:
            terms.append("ACE inhibitor ARB CKD kidney protection")
        if "blood pressure" in case_lower or "hypertension" in case_lower:
            terms.append("blood pressure target CKD hypertension")
        if "sglt2" in case_lower:
            terms.append("SGLT2 inhibitor CKD progression")
        if terms:
            return " ".join(terms)
        return "chronic kidney disease CKD eGFR management treatment proteinuria"

    def analyze(self, clinical_case: str) -> dict:
        print(f"[{self.name}] Analyzing case...")
        rag_query = self._get_rag_query(clinical_case)
        retrieved_docs = self.retriever.invoke(rag_query)
        web_evidence = fetch_pubmed_evidence(self._get_pubmed_query(clinical_case))
        if web_evidence:
            print(f"[{self.name}] PubMed evidence fetched ✓")
        augmented_case = clinical_case + web_evidence
        answer = self.chain.invoke(augmented_case)
        return {
            "agent": self.name, "specialty": self.specialty,
            "analysis": answer, "web_evidence_used": bool(web_evidence),
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
