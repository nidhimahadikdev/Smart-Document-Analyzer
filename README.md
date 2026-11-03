# Smart Document Analyzer 📄🤖

An intelligent document analysis and RAG (Retrieval-Augmented Generation) application powered by multi-agent workflows, vector search, and local LLMs.

---

## 🚀 Overview

Smart Document Analyzer enables users to upload PDF documents, extract information, perform intelligent document QA, analyze risks and compliance, summarize contents, and compare multiple documents.

### ✨ Key Features
- **Multi-Agent Routing System**: Automatically routes user queries to specialized agents:
  - 🔍 **RAG Agent**: Document search & contextual Q&A with source attribution
  - ⚠️ **Risk Agent**: Analyzes risks, penalties, and compliance obligations
  - 📝 **Summary Agent**: Generates structured executive summaries
  - ⚖️ **Comparison Agent**: Analyzes differences across multiple documents
  - 💬 **General Agent**: Handles general interactions & greetings
- **Vector Search & Ingestion**:
  - Semantic vector storage with ChromaDB & FAISS
  - Hybrid search retrieval with BM25 keyword matching
  - Efficient PDF document parsing using PyMuPDF (`fitz`)
- **Interactive UI**:
  - Built with Streamlit with a modern dark theme
  - Export analysis reports to PDF with ReportLab
  - Model evaluation dashboard with Plotly charts

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **LLM / Orchestration**: LangChain, ChatOllama (`qwen2.5`)
- **Vector Store**: ChromaDB, FAISS
- **Retrievers**: BM25, Ensemble Hybrid Retriever
- **Document Parsing**: PyMuPDF
- **Visualization**: Plotly
- **Reporting**: ReportLab

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nidhimahadikdev/Smart-Document-Analyzer.git
cd Smart-Document-Analyzer
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys if needed:
```bash
cp .env.example .env
```

### 5. Run the Application
```bash
streamlit run app.main.py
```

---

## 📊 Evaluation & Dashboards

- Run `model_evaluation.py` to evaluate model metrics.
- Run `dashboard_eval.py` or inspect `Model_Evaluation_Dashboard.html` for model benchmark visualizations.
