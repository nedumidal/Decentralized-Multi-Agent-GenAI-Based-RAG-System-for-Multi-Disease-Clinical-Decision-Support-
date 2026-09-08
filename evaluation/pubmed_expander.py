"""
PubMed Knowledge Base Expander — Step 4
Downloads recent PubMed articles per specialty and adds
them to existing FAISS vectorstores to improve retrieval recall.

Base paper used: 889 PubMed articles + 5,630 patient texts
Our target:      50-100 articles per specialty (200-400 total)
"""

import os
import time
import json
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ── BioBERT embeddings ────────────────────────────────────────────────────────
def get_embeddings():
    print("Loading BioBERT embeddings...")
    return HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ── PubMed search queries per specialty ───────────────────────────────────────
SPECIALTY_QUERIES = {
    "cardiology": [
        "heart failure with reduced ejection fraction treatment 2024",
        "HFrEF guideline-directed medical therapy GDMT",
        "SGLT2 inhibitor heart failure outcomes randomized trial",
        "sacubitril valsartan heart failure treatment",
        "beta blocker ACE inhibitor heart failure mortality"
    ],
    "nephrology": [
        "chronic kidney disease CKD management guidelines 2024",
        "SGLT2 inhibitor CKD progression prevention",
        "ACE inhibitor ARB proteinuria reduction kidney",
        "eGFR monitoring CKD treatment outcomes",
        "finerenone chronic kidney disease cardiovascular"
    ],
    "diabetology": [
        "type 2 diabetes SGLT2 inhibitor heart failure kidney",
        "GLP-1 receptor agonist semaglutide cardiovascular outcomes",
        "HbA1c target diabetes treatment guidelines 2024 2025",
        "metformin CKD contraindication dose adjustment",
        "diabetes comorbidity heart failure treatment"
    ],
    "pharmacology": [
        "drug interaction heart failure diabetes kidney disease",
        "renal dosing adjustment medications eGFR impairment",
        "furosemide loop diuretic heart failure clinical trial",
        "enalapril lisinopril heart failure kidney dose",
        "polypharmacy cardiovascular renal diabetes management"
    ]
}

# ── Fetch PubMed abstracts via E-utilities API ────────────────────────────────
def search_pubmed(query: str, max_results: int = 20) -> list:
    """Search PubMed and return list of PMIDs"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "datetype": "pdat",
        "mindate": "2020",
        "maxdate": "2025"
    }

    try:
        resp = requests.get(search_url, params=params, timeout=10)
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    except Exception as e:
        print(f"  Search error: {e}")
        return []

def fetch_abstracts(pmids: list) -> list:
    """Fetch abstracts for given PMIDs"""
    if not pmids:
        return []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text"
    }

    try:
        resp = requests.get(fetch_url, params=params, timeout=15)
        text = resp.text
        # Split by PMID markers
        articles = []
        blocks = text.split("\n\n\n")
        for block in blocks:
            block = block.strip()
            if len(block) > 100:  # Skip empty blocks
                articles.append(block)
        return articles
    except Exception as e:
        print(f"  Fetch error: {e}")
        return []

# ── Add articles to FAISS vectorstore ────────────────────────────────────────
def expand_vectorstore(specialty: str,
                       articles: list,
                       embeddings) -> int:
    """Add new articles to existing BioBERT vectorstore"""
    faiss_path = f"vectorstores_biobert/{specialty}"

    if not os.path.exists(faiss_path):
        print(f"  No existing vectorstore for {specialty} — skipping")
        return 0

    # Load existing vectorstore
    vectorstore = FAISS.load_local(
        faiss_path, embeddings,
        allow_dangerous_deserialization=True
    )

    # Prepare documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )

    docs = []
    for i, article in enumerate(articles):
        doc = Document(
            page_content=article,
            metadata={
                "source": f"pubmed_{specialty}_{i}",
                "specialty": specialty,
                "type": "pubmed_abstract"
            }
        )
        docs.append(doc)

    if not docs:
        return 0

    chunks = splitter.split_documents(docs)
    print(f"  Adding {len(chunks)} new chunks to {specialty} vectorstore...")

    # Add to existing vectorstore
    vectorstore.add_documents(chunks)

    # Save updated vectorstore
    vectorstore.save_local(faiss_path)
    print(f"  ✅ {specialty} vectorstore expanded with {len(chunks)} chunks")

    return len(chunks)

# ── Main expander ─────────────────────────────────────────────────────────────
def expand_all_knowledge_bases(articles_per_query: int = 15):
    print("\n" + "="*60)
    print("MARC-Clinical PubMed Knowledge Base Expander")
    print("="*60)

    embeddings = get_embeddings()
    summary = {}

    for specialty, queries in SPECIALTY_QUERIES.items():
        print(f"\n📚 Expanding {specialty} knowledge base...")
        all_articles = []

        for query in queries:
            print(f"  Searching: '{query[:50]}...'")
            pmids = search_pubmed(query, max_results=articles_per_query)
            print(f"  Found {len(pmids)} PMIDs")

            if pmids:
                abstracts = fetch_abstracts(pmids)
                all_articles.extend(abstracts)
                print(f"  Fetched {len(abstracts)} abstracts")

            # Respect NCBI rate limit (3 requests/second)
            time.sleep(0.4)

        # Remove duplicates
        unique_articles = list(set(all_articles))
        print(f"  Total unique articles: {len(unique_articles)}")

        # Expand vectorstore
        chunks_added = expand_vectorstore(
            specialty, unique_articles, embeddings
        )
        summary[specialty] = {
            "articles_fetched": len(unique_articles),
            "chunks_added": chunks_added
        }

    # Save summary
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/pubmed_expansion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("Knowledge Base Expansion Complete")
    print("="*60)
    for specialty, stats in summary.items():
        print(f"  {specialty}: {stats['articles_fetched']} articles, "
              f"{stats['chunks_added']} chunks added")
    print("\nRun python main.py to use expanded knowledge bases")
    print("="*60)

    return summary

if __name__ == "__main__":
    expand_all_knowledge_bases(articles_per_query=15)
