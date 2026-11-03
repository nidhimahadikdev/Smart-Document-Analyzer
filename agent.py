import json
import re
from langchain_ollama import ChatOllama

# Offline Qwen 2.5 Model
llm = ChatOllama(model="qwen2.5", temperature=0.3, keep_alive=-1)

def route_query(query):
    try:
        prompt = f"""
        Classify the user's query into EXACTLY ONE word: SUMMARY, RISK, COMPARISON, RAG, or GENERAL.
        
        Strict Rules:
        - RISK: The user asks about risks, penalties, or compliance.
        - COMPARISON: The user wants to compare documents.
        - GENERAL: ONLY simple greetings ("hi", "hello").
        - SUMMARY: The user EXPLICITLY asks to "write a summary" or "summarize the document".
        - RAG: EVERYTHING ELSE. If the user asks "tell me about my document", questions about the document, or anything else, route to RAG.
        
        User Query: "{query}"
        """
        response = llm.invoke(prompt)
        decision = response.content.strip().upper()
        
        for valid_category in ["SUMMARY", "RISK", "COMPARISON", "RAG", "GENERAL"]:
            if valid_category in decision:
                return valid_category
        return "RAG" 
    except Exception:
        return "RAG"

def run_general_agent(query):
    try:
        prompt = f"You are a helpful AI Assistant. Answer this general query or greeting directly: '{query}'"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ **Local Connection Error:** {str(e)}"

def run_rag_agent(query, retriever):
    try:
        if "k" in retriever.search_kwargs:
            retriever.search_kwargs["k"] = 3
            
        docs = retriever.invoke(query)
        if not docs:
            return "I could not find relevant information in the uploaded documents.", []
            
        context = "\n\n".join([f"Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})\nText: {d.page_content}" for d in docs])
        
        prompt = f"""
        You are an intelligent document assistant. 
        First, look at the context below to see how it relates to the user's question. 
        If the context only mentions a topic but doesn't explain what it is, use your general knowledge to provide a helpful, detailed explanation of that topic.
        
        Context:\n{context}\n\nQuestion: {query}
        """
        response = llm.invoke(prompt)
        return response.content, docs
    except Exception as e:
        return f"⚠️ **Error:** {str(e)}", []

def run_risk_agent(retriever):
    try:
        if "k" in retriever.search_kwargs:
            retriever.search_kwargs["k"] = 30
            
        docs = retriever.invoke("risks obligations penalties liability strict deadlines")
        context = "\n\n".join([f"Document: {d.metadata.get('source')}\nText: {d.page_content}" for d in docs])
        word_count = len(context.split())
        
        prompt = f"""
        Analyze the context for risks. Return ONLY valid JSON matching this structure exactly:
        {{"risk_score": 50, "flags_found": 2, "risk_assessment": "Details here"}}
        
        IMPORTANT: In the "risk_assessment" field, if there are multiple documents, separate them clearly using explicit text like "DOCUMENT: filename.pdf". 
        Do NOT use raw line breaks that break JSON formatting.
        
        Context: {context}
        """
        response = llm.invoke(prompt)
        raw_text = response.content.strip()
        
        risk_data = None
        
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                clean_json = raw_text[start_idx:end_idx+1]
                risk_data = json.loads(clean_json, strict=False)
            except json.JSONDecodeError:
                pass 
                
        if not risk_data:
            score_match = re.search(r'risk[_\s]?score[^\d]*(\d+)', raw_text, re.IGNORECASE)
            flags_match = re.search(r'flags?[_\s]?found[^\d]*(\d+)', raw_text, re.IGNORECASE)
            
            score = int(score_match.group(1)) if score_match else 35 
            flags = int(flags_match.group(1)) if flags_match else 1
            
            risk_data = {
                "risk_score": score,
                "flags_found": flags,
                "risk_assessment": raw_text 
            }
            
        risk_data["word_count"] = word_count 
        return risk_data
            
    except Exception as e:
        return {"risk_score": 0, "flags_found": 0, "word_count": 0, "risk_assessment": f"Parsing Failed: {str(e)}"}

def run_summary_agent(retriever, summary_type="Detailed Summary"):
    try:
        if "k" in retriever.search_kwargs:
            retriever.search_kwargs["k"] = 15  
            
        docs = retriever.invoke("executive summary main points introduction")
        context = "\n\n".join([f"Document: {d.metadata.get('source')}\nText: {d.page_content}" for d in docs])
        
        if summary_type == "Short Summary":
            instruction = "Write a brief, 3-4 sentence short summary."
        elif summary_type == "Key Points":
            instruction = "Extract the main points as a concise bulleted list. Do not write long paragraphs, just short, clear bullet points."
        elif summary_type == "Action Items":
            instruction = "List any actionable tasks, requirements, or next steps found in the document as short bullet points."
        else:
            instruction = "Write a comprehensive, detailed summary."
        
        prompt = f"""
        {instruction}
        
        CRITICAL INSTRUCTION: If the context contains information from multiple different documents, you MUST provide a clearly separated section for EACH document. Use the Document Name as a bold heading.
        
        Document Content:
        {context}
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ **Error:** {str(e)}"

def run_comparison_agent(retriever, query):
    try:
        # Increase k to ensure we pull enough text from BOTH documents
        if "k" in retriever.search_kwargs:
            retriever.search_kwargs["k"] = 15
            
        # If the query is just "compare these documents", we pull general overview chunks
        search_query = query if len(query) > 25 else "introduction main topics overview concepts"
        docs = retriever.invoke(search_query)
        
        context = "\n\n".join([f"Document: {d.metadata.get('source')}\nText: {d.page_content}" for d in docs])
        
        # STRICT COMPARISON PROMPT: Forces the AI to compare whatever is in the context without asking for permission
        prompt = f"""
        You are a Document Comparison Agent.
        User's Request: '{query}'
        
        Based ONLY on the context below, provide a detailed comparison between the documents. 
        If the user did not specify exactly what to compare, automatically provide a general overview of their similarities and differences.
        DO NOT ask the user which documents to compare or what to compare. Just compare the ones provided in the Context!
        
        Context:
        {context}
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ **Error:** {str(e)}"