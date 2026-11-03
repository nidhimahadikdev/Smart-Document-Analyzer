import warnings
warnings.filterwarnings("ignore")

import os
import io
import streamlit as st

# Must be the very first Streamlit command executed
st.set_page_config(page_title="Smart Document Analyzer", page_icon="📄", layout="wide")

import plotly.graph_objects as go
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from document_processing import process_and_store_documents, get_retriever
import agent

load_dotenv()

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stat-box {
        background-color: #1E293B;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #334155;
    }
    .agent-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        background-color: #1E3A8A;
        color: #93C5FD;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "risk_data" not in st.session_state:
    st.session_state.risk_data = None
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []

def generate_pdf_report(risk_data, summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>Smart Document Analyzer — Executive Dossier</b>", styles["Heading1"]))
    story.append(Spacer(1, 14))

    if risk_data:
        story.append(Paragraph(f"<b>Overall Risk Score:</b> {risk_data.get('risk_score', 0)} / 100", styles["Normal"]))
        story.append(Paragraph(f"<b>Flags Identified:</b> {risk_data.get('flags_found', 0)}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Risk & Clause Assessment:</b>", styles["Heading2"]))
        clean_assessment = risk_data.get("risk_assessment", "").replace("\n", "<br/>")
        story.append(Paragraph(clean_assessment, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

def render_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 20, 'color': "#F8FAFC"}},
        number={'font': {'size': 44, 'color': "#F8FAFC"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
            'bar': {'color': "#22C55E" if score < 40 else ("#EAB308" if score < 75 else "#EF4444")},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.15)"},
                {'range': [40, 75], 'color': "rgba(234, 179, 8, 0.15)"},
                {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.15)"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"}, height=280, margin=dict(l=20, r=20, t=40, b=10))
    return fig

with st.sidebar:
    st.title("Control Panel")
    st.success("System Online: Local Qwen 2.5 Active.")
    st.markdown("---")
    st.subheader("1. Ingestion Pipeline")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Document(s)", use_container_width=True):
        with st.spinner("Executing Ingestion Pipeline into ChromaDB..."):
            success, msg = process_and_store_documents(uploaded_files)
            if success:
                st.session_state.processed = True
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    st.subheader("2. Executive Actions")
    run_risk_btn = st.button("Run Risk Assessment", use_container_width=True)
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_trace = []
        st.rerun()

st.title("Smart Document Analyzer")
st.caption("Agentic AI + RAG-Based Document Intelligence System")

tab_qa, tab_risk, tab_summary, tab_trace = st.tabs(["💬 Conversational RAG", "📊 Risk Assessment", "📝 Synthesis & Summary", "🤖 Agent Activity Trace"])

with tab_qa:
    chat_container = st.container()
    user_query = st.chat_input("Ask a question, say hello, request risks, or request a summary...")

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                try:
                    decision = agent.route_query(user_query)
                    st.session_state.agent_trace.append(f"Query: '{user_query}' -> Orchestrator selected: **{decision} AGENT**")

                    with st.spinner(f"Invoking {decision} Agent..."):
                        pill_html = ""
                        if "GENERAL" in decision:
                            answer = agent.run_general_agent(user_query)
                            pill_html = "<span class='agent-pill'>Handled by: GENERAL Agent</span><br><br>"
                        elif not st.session_state.processed:
                            answer = "Please upload and process a document in the Control Panel first so I can analyze it for you."
                        else:
                            retriever = get_retriever()
                            if decision == "SUMMARY":
                                answer = agent.run_summary_agent(retriever)
                            elif decision == "RISK":
                                risk_res = agent.run_risk_agent(retriever)
                                st.session_state.risk_data = risk_res
                                answer = f"**Risk Assessment Completed.** Risk Score: {risk_res.get('risk_score')}/100. Check the **Risk Assessment** tab."
                            elif decision == "COMPARISON":
                                answer = agent.run_comparison_agent(retriever, user_query)
                            else:
                                answer, docs = agent.run_rag_agent(user_query, retriever)
                            pill_html = f"<span class='agent-pill'>Handled by: {decision} Agent</span><br><br>"
                        
                        final_answer = pill_html + answer
                        st.markdown(final_answer, unsafe_allow_html=True)
                        
                except Exception as e:
                    final_answer = f"⚠️ **System Error:** {str(e)}"
                    st.error(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

with tab_risk:
    if run_risk_btn or st.session_state.risk_data:
        if not st.session_state.processed:
            st.warning("Upload and process documents before running Risk Assessment.")
        else:
            if run_risk_btn:
                with st.spinner("Risk Agent scanning clauses and liabilities..."):
                    retriever = get_retriever()
                    st.session_state.risk_data = agent.run_risk_agent(retriever)
            
            risk = st.session_state.risk_data
            if risk:
                col1, col2, col3 = st.columns(3)
                with col1: st.markdown(f"<div class='stat-box'><h3>Risk Score</h3><h1>{risk.get('risk_score', 0)}</h1></div>", unsafe_allow_html=True)
                with col2: st.markdown(f"<div class='stat-box'><h3>Flags</h3><h1>{risk.get('flags_found', 0)}</h1></div>", unsafe_allow_html=True)
                with col3: st.markdown(f"<div class='stat-box'><h3>Words</h3><h1>{risk.get('word_count', 0)}</h1></div>", unsafe_allow_html=True)
                
                st.plotly_chart(render_risk_gauge(risk.get("risk_score", 0)), use_container_width=True)
                st.markdown(risk.get("risk_assessment", ""))
    else:
        st.info("Click 'Run Risk Assessment' in the sidebar.")

with tab_summary:
    summary_mode = st.radio("Select Mode:", ["Short Summary", "Detailed Summary", "Key Points", "Action Items"], horizontal=True)
    if st.button("Generate Summary"):
        if not st.session_state.processed:
            st.warning("Please upload and process documents first.")
        else:
            with st.spinner(f"Producing {summary_mode}..."):
                st.markdown(agent.run_summary_agent(get_retriever(), summary_type=summary_mode))

with tab_trace:
    if not st.session_state.agent_trace:
        st.info("No queries executed yet.")
    else:
        for idx, event in enumerate(st.session_state.agent_trace, 1):
            st.markdown(f"**Step {idx}:** {event}")