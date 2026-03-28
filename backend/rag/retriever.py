#This code loads your FAISS vector database and converts a user query into embeddings to search for the most relevant text chunks.

#It returns the top matching content (with sources) as context, which is then used by your AI to generate accurate answer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_DIR = os.path.join(_BACKEND_ROOT, "faiss_index")

_vectorstore = None
_load_error_logged = False

def get_embeddings():
    """Load the HuggingFace model. Runs on your CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vectorstore():
    """Load FAISS index from disk. Only loads once per server start."""
    global _vectorstore
    if _vectorstore is None:
        if not os.path.isdir(FAISS_INDEX_DIR):
            raise FileNotFoundError(
                f"faiss_index not found at {FAISS_INDEX_DIR}. From backend folder run: python -m rag.ingest"
            )
        print(f"Loading FAISS index from {FAISS_INDEX_DIR}...")
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully!")
    return _vectorstore

def retrieve_relevant_context(query: str, k: int = 5) -> str:
    """
    Search FAISS for chunks most relevant to the query.
    
    query = the user's question or search text
    k     = how many chunks to return (default 5)
    
    Returns all chunks joined as one big string.
    If the index is missing or load/search fails, returns empty string so API routes still work.
    """
    global _load_error_logged
    try:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        if not _load_error_logged:
            print(f"[RAG] Knowledge search unavailable ({e!s}). Continuing without retrieved context.")
            _load_error_logged = True
        return ""

    context_parts = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "knowledge base"))
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")

    return "\n\n".join(context_parts)