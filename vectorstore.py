import os
import pickle
import logging
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configure logging to show the AI's thought process in your terminal during the presentation
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def create_vector_store(chunks):
    """
    Creates both a Dense (Semantic) and Sparse (Keyword) database.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        
        bm25_retriever = BM25Retriever.from_documents(chunks)
        with open("bm25_retriever.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        return True
    except Exception as e:
        print(f"Error creating hybrid vector store: {e}")
        return False

@st.cache_resource
def get_conversational_chain():
    prompt_template = """
    You are an advanced AI assistant and Expert Academic Analyzer. 
    You have access to both the user's uploaded document context and the internet.

    Strict Routing Rules:
    1. GREETINGS: If the user says "hello", "hi", etc., greet them politely.
    2. IN-DOCUMENT QUESTIONS: If the question can be answered by the Context, answer it using clear bullet points and **bold text**.
    3. OUT-OF-DOCUMENT: If the specific information is NOT in the Context, you must output exactly this secret trigger code and nothing else: "ROUTING_TO_WEB_SEARCH"

    Context:
    {context}

    Conversation History:
    {chat_history}

    Question: 
    {question}

    Answer:
    """
    # FIXED: Reverted to the highly stable gemini-1.5-flash
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_rag_response(user_question, chat_history):
    """
    Retrieves chunks using Multi-Query Expansion wrapped around a Hybrid Search Ensemble.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # FIXED: Reverted to the highly stable gemini-1.5-flash
        llm_for_retrieval = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Load Hybrid Retrievers
        faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
        
        with open("bm25_retriever.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 3
        
        # Create Hybrid Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
        )
        
        # AGENTIC UPGRADE: Wrap the Ensemble in a Multi-Query Generator
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever, 
            llm=llm_for_retrieval
        )
        
        # Execute Expanded Retrieval
        docs = advanced_retriever.invoke(user_question)
        
        formatted_history = ""
        for message in chat_history:
            formatted_history += f"{message['role'].capitalize()}: {message['content']}\n"
        
        chain = get_conversational_chain()
        response = chain.invoke(
            {
                "input_documents": docs,
                "chat_history": formatted_history,
                "question": user_question
            },
            return_only_outputs=True
        )
        
        answer = response["output_text"]
        
        if "ROUTING_TO_WEB_SEARCH" in answer:
            search = DuckDuckGoSearchRun()
            web_results = search.run(user_question)
            return f"**System Alert:** Information not found in the uploaded PDF. Automatically routing to Live Web Search...\n\n**Live Search Results:**\n{web_results}"
        
        is_greeting = answer.lower().startswith(("hello", "hi", "greetings", "welcome"))
        
        if not is_greeting:
            sources = set()
            for doc in docs:
                if "page" in doc.metadata:
                    sources.add(doc.metadata["page"] + 1)
            
            if sources:
                source_pages = ", ".join(map(str, sorted(sources)))
                answer += f"\n\n**Sources:** Found on Page(s) {source_pages}\n*(Retrieval Engine: Multi-Query Expansion + Hybrid Ensemble)*"
            
        return answer
    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "**System Alert:** The API rate limit has been reached. Please wait a few moments."
        else:
            return f"**System Alert:** An unexpected processing error occurred. Details: {error_msg}"