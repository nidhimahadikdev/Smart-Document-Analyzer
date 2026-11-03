import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf_documents(uploaded_files):
    """
    Extracts text from uploaded PDF files and preserves page metadata for accurate RAG citations.
    """
    documents = []
    
    # Handle both single file and multiple files
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            if text.strip():
                # RAG Metadata: Crucial for telling the user exactly where the answer came from
                metadata = {
                    "source": uploaded_file.name,
                    "page": page_num + 1
                }
                documents.append(Document(page_content=text, metadata=metadata))
                
    return documents

def split_text_into_chunks(documents):
    """
    Splits documents into overlapping semantic chunks to optimize Vector Database retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Keeps paragraphs and sentences together
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks