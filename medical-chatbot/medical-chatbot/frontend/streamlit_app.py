import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = "http://localhost:5000"

st.set_page_config(
    page_title="MedBot - Your Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical UI
st.markdown("""
    <style>
    body {
        background-color: #f3f6fa;
    }
    .main {
        background-color: #f9fbfd;
        padding: 1rem;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        color: #1b1f23;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #ffffff;
        border-radius: 6px;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e7f3fe;
        border-left: 5px solid #0077b6;
    }
    .chat-message.bot {
        background-color: #f1fcfc;
        border-left: 5px solid #00b4d8;
    }
    .message {
        font-size: 1rem;
        line-height: 1.6;
    }
    .sources {
        font-size: 0.8rem;
        color: #555;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .sidebar .sidebar-content {
        background-color: #0077b6;
        color: white;
    }
    h1, h2, h3 {
        color: #023e8a;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.image("https://cdn-icons-png.flaticon.com/512/3771/3771553.png", width=100)
st.title("MedBot: Your AI Medical Assistant")
st.markdown("💬 *Get answers from trusted medical sources. Designed for clarity, reliability, and care.*")

# Sidebar
with st.sidebar:
    st.header("🔎 About MedBot")
    st.markdown("""
    MedBot leverages **RAG (Retrieval-Augmented Generation)** and **AI models** to answer medical questions from verified documents.

    🧠 *Built with a knowledge-first approach.*  
    ⚠️ *Not a replacement for a licensed doctor.*
    """)

    st.divider()
    st.subheader("🛠️ Admin Panel")

    # PDF processing
    with st.expander("📄 Process Medical PDFs"):
        pdf_dir = st.text_input("Enter PDF Folder Path", "/path/to/pdfs")
        if st.button("Start Processing"):
            if pdf_dir:
                with st.spinner("Processing PDFs..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/process_pdfs", json={"directory": pdf_dir})
                        if res.status_code == 202:
                            st.success("✅ PDFs are being processed.")
                        else:
                            st.error(f"❌ Error: {res.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"🚫 Connection error: {str(e)}")

    # System status
    with st.expander("📡 Check System Status"):
        if st.button("Check"):
            try:
                res = requests.get(f"{BACKEND_URL}/status")
                if res.status_code == 200:
                    st.success("✅ System is ready.")
                elif res.status_code == 202:
                    st.info("⌛ System is still initializing...")
                else:
                    st.error(f"❌ Error: {res.json().get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"🚫 Connection error: {str(e)}")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display
st.subheader("💬 Ask a Medical Question")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message">🙋 <b>You:</b> {msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            sources_html = "<div class='sources'>Sources: " + ", ".join(msg["sources"]) + "</div>"
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="message">🩺 <b>MedBot:</b> {msg["content"]}</div>
            {sources_html}
        </div>
        """, unsafe_allow_html=True)

# Chat input
user_query = st.text_area("Type your question:", key="user_input", height=100)

if st.button("Ask"):
    if user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("MedBot is thinking..."):
            try:
                status = requests.get(f"{BACKEND_URL}/status")
                if status.status_code != 200:
                    resp_text = "System is not ready. Please try again later." if status.status_code == 202 else "Unknown error."
                    sources = []
                else:
                    res = requests.post(f"{BACKEND_URL}/query", json={"query": user_query})
                    if res.status_code == 200:
                        data = res.json()
                        resp_text = data["response"]
                        sources = data.get("sources", [])
                    else:
                        resp_text = f"❌ Error: {res.json().get('error', 'Unknown error')}"
                        sources = []
            except Exception as e:
                resp_text = f"⚠️ Connection error: {str(e)}"
                sources = []

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": resp_text,
            "sources": sources
        })
        st.rerun()


# Footer disclaimer
st.markdown("---")
st.caption("""
⚠️ **Disclaimer:** MedBot is for informational purposes only and does not replace medical consultation.  
Always consult a licensed professional for health concerns.
""")
