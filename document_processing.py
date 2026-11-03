import os
import shutil
import pymupdf  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
import chromadb  # Added to access the system cache

CHROMA_PATH = "chroma_db_final"

def load_pdf_documents(uploaded_files):
    documents = []
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")  
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                metadata = {"source": uploaded_file.name, "page": page_num + 1}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents

def process_and_store_documents(uploaded_files):
    # Fix: Clear ChromaDB's internal memory cache before deleting the files
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    documents = load_pdf_documents(uploaded_files)
    if not documents:
        return False, "No readable text found in the documents."
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    
    # Offline Embeddings (No internet required)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    return True, f"Successfully processed {len(chunks)} chunks locally."

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})