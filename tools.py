# tools.py
# Defines the tools available to the AI Agent.

def document_search_tool(query):
    """
    Tool for the Agent to access the FAISS vector database.
    (Core execution is currently routed through vectorstore.py for UI stability).
    """
    return "Searching document database for: " + query

def general_chat_tool(query):
    """
    Fallback tool for general conversation not requiring the document.
    """
    return "Processing general query: " + query