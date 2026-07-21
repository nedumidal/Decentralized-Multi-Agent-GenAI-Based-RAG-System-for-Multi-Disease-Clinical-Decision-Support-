import os
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
        print("Loading embedding model (one time only)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding model loaded.")
    return _embeddings


class PharmacologyAgent:
    def __init__(self, pdf_path: str):
        self.name = "Pharmacology Agent"
        self.specialty = "pharmacology"
        self.retriever = None
        self.chain = None
        self._build_knowledge_base(pdf_path)

    def _build_knowledge_base(self, pdf_path: str):
        print(f"[{self.name}] Loading knowledge base...")
        faiss_path = f"vectorstores/{self.specialty}"
        embeddings = get_embeddings()

        if os.path.exists(faiss_path):
            print(f"[{self.name}] Loading saved vectorstore from disk...")
            vectorstore = FAISS.load_local(
                faiss_path, embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[{self.name}] Vectorstore loaded instantly.")
        else:
            print(f"[{self.name}] Building vectorstore from PDF...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(documents)
            print(f"[{self.name}] Split into {len(chunks)} chunks.")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(faiss_path, exist_ok=True)
            vectorstore.save_local(faiss_path)
            print(f"[{self.name}] Vectorstore saved to disk.")

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        print(f"[{self.name}] Knowledge base ready.")

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )

        prompt = PromptTemplate.from_template("""You are a specialist Pharmacologist AI agent.
Analyze this clinical case using WHO Essential Medicines guidelines context provided.
Extract ALL relevant drug recommendations, dosing, contraindications and interactions.
Cite specific sections and page numbers.
Focus on: drug interactions between heart failure medications and diabetes medications,
dosing adjustments for reduced kidney function (eGFR 42), contraindicated drugs,
safe medication combinations, monitoring requirements.
Even if context is partial, extract what is available.

Context: {context}

Clinical Case: {question}

Pharmacology Analysis (with citations):""")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def _get_focused_query(self) -> str:
        return "heart failure diabetes kidney disease drug interactions medications dosing adjustment renal impairment contraindications diuretics ACE inhibitor"

    def analyze(self, clinical_case: str) -> dict:
        print(f"[{self.name}] Analyzing case...")
        focused_query = self._get_focused_query()
        retrieved_docs = self.retriever.invoke(focused_query)
        answer = self.chain.invoke(clinical_case)

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "analysis": answer,
            "retrieved_docs": [
                {
                    "doc_id": doc.metadata.get("page", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:150]
                }
                for doc in retrieved_docs
            ],
            "confidence": self._estimate_confidence(answer)
        }

    def _estimate_confidence(self, answer: str) -> float:
        if "Not found" in answer and len(answer) < 100:
            return 0.2
        elif "recommend" in answer.lower() or "guideline" in answer.lower():
            return 0.85
        return 0.65