import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from pypdf import PdfReader

from utils.memory import (
    new_session_id, list_sessions, load_session,
    save_session, delete_session, session_title
)
from utils.rag import chunk_text, build_index, retrieve

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Siddhi AI Pro", page_icon="🤖", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 0%, #1e293b 0%, #0f172a 45%, #060a14 100%);
    color: #e2e8f0;
}
#MainMenu, footer { visibility: hidden; }

.main-title {
    text-align:center; font-size:46px; font-weight:800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 0;
}
.subtitle {
    text-align:center; color:#94a3b8; font-size:16px;
    margin-top:4px; margin-bottom:28px;
}
.section-label {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #64748b; margin: 18px 0 10px 2px;
}
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px; padding:14px 16px; margin-top:12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.stButton button {
    width:100%; border-radius:12px;
    border: 1px solid rgba(255,255,255,0.1);
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12));
    color: #e2e8f0; font-weight: 600; padding: 8px 0;
    transition: all 0.2s ease;
}
.stButton button:hover {
    border-color: #38bdf8;
    background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(129,140,248,0.25));
    transform: translateY(-1px);
}
[data-testid="stSidebar"] {
    background: #0b1120; border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 10px 14px;
}
hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ---------------- API ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error("⚠ GROQ_API_KEY not found. Set it in .env or Streamlit secrets.")
    st.stop()
client = Groq(api_key=api_key)


def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        st.sidebar.error(f"PDF read error: {e}")
        return ""


def stream_reply(response):
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------- SESSION STATE ----------------
if "session_id" not in st.session_state:
    existing = list_sessions()
    st.session_state.session_id = existing[0] if existing else new_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = load_session(st.session_state.session_id)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None  # (vectorizer, matrix, chunks)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙ Control Panel")

    if st.button("➕ New Chat"):
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.session_state.rag_index = None
        st.rerun()

    st.markdown('<p class="section-label">Chat History</p>', unsafe_allow_html=True)
    for sid in list_sessions():
        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button(session_title(sid), key=f"sel_{sid}"):
                st.session_state.session_id = sid
                st.session_state.messages = load_session(sid)
                st.session_state.rag_index = None
                st.rerun()
        with c2:
            if st.button("🗑", key=f"del_{sid}"):
                delete_session(sid)
                if sid == st.session_state.session_id:
                    st.session_state.session_id = new_session_id()
                    st.session_state.messages = []
                st.rerun()

    st.markdown("---")
    personality = st.selectbox(
        "AI Personality",
        ["Coding Mentor", "Interview Coach", "Study Buddy", "General Assistant"]
    )

    uploaded_pdfs = st.file_uploader(
        "📄 Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded_pdfs:
        all_chunks = []
        for pdf in uploaded_pdfs:
            text = extract_pdf_text(pdf)
            all_chunks.extend(chunk_text(text))
        vectorizer, matrix = build_index(all_chunks)
        st.session_state.rag_index = (vectorizer, matrix, all_chunks)
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_pdfs)} file(s)")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Model", "Llama 3.3")
    c2.metric("Status", "🟢 Online")

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🤖 Siddhi AI Pro</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">RAG-powered AI assistant with memory, streaming & multi-chat</p>',
    unsafe_allow_html=True
)

# ---------------- QUICK ACTIONS ----------------
st.markdown('<p class="section-label">Quick Actions</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💻 Debug Code"):
        st.session_state.quick_prompt = "Help me debug my code"
with col2:
    if st.button("📚 Explain DSA"):
        st.session_state.quick_prompt = "Explain binary search"
with col3:
    if st.button("🎯 Career Advice"):
        st.session_state.quick_prompt = "How to become AI engineer?"

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# ---------------- INPUT ----------------
prompt = st.chat_input("Ask anything...")
if not prompt and "quick_prompt" in st.session_state:
    prompt = st.session_state.quick_prompt
    del st.session_state.quick_prompt

system_prompt = {
    "Coding Mentor": "You are an expert coding mentor.",
    "Interview Coach": "You are a placement interview coach.",
    "Study Buddy": "You explain difficult topics simply.",
    "General Assistant": "You are a helpful AI assistant."
}

# ---------------- RESPONSE ----------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_session(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)

    context = system_prompt[personality]

    if st.session_state.rag_index:
        vectorizer, matrix, chunks = st.session_state.rag_index
        relevant = retrieve(prompt, vectorizer, matrix, chunks, top_k=3)
        if relevant:
            context += "\nRelevant document context:\n" + "\n---\n".join(relevant)

    start = time.time()
    reply = ""

    with st.chat_message("assistant", avatar="🤖"):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": context},
                    *st.session_state.messages
                ],
                stream=True
            )
            reply = st.write_stream(stream_reply(response))
        except Exception as e:
            reply = f"⚠ Something went wrong: {e}"
            st.write(reply)

        end = time.time()
        st.caption(f"Response time: {round(end - start, 2)} sec")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_session(st.session_state.session_id, st.session_state.messages)

# ---------------- EXPORT ----------------
chat_export = json.dumps(st.session_state.messages, indent=2)
st.download_button(
    "📥 Download Chat History",
    chat_export,
    file_name="chat_history.json",
    mime="application/json"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(f"Built by Siddhi Varshney | {datetime.now().strftime('%d %b %Y')}")