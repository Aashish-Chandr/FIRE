#This script reads all .txt files, splits them into smaller chunks, converts them into vector embeddings, and stores them in a FAISS index
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE_DIR = os.path.join(_RAG_DIR, "Knowledge_base")
_FAISS_OUT = os.path.join(os.path.dirname(_RAG_DIR), "faiss_index")


def build_faiss_index():
    print("Step 1: Loading .txt files from knowledge_base folder...")

    loader = DirectoryLoader(
        _KNOWLEDGE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} documents")

    print("Step 2: Splitting documents into small chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      
        chunk_overlap=50    
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3: Loading HuggingFace embedding model...")
    print("  (First time: downloads ~90MB model. Be patient!)")
 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},              
        encode_kwargs={"normalize_embeddings": True} 
    )

    print("Step 4: Building FAISS index...")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Step 5: Saving FAISS index to disk...")

    vectorstore.save_local(_FAISS_OUT)

    print(f"\nDone! FAISS index saved to {_FAISS_OUT}")
    print("Now run: uvicorn main:app --reload (from the backend directory)")


if __name__ == "__main__":
    build_faiss_index()