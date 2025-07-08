import streamlit as st
from qabot import retriever_qa
from datetime import datetime
import json
import os

# --- Constants ---
CHAT_LOG_FILE = "chat_logs.json"


# --- Helpers ---
def load_chat_logs():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_chat_logs(logs):
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


# --- Page config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")

# --- Title + Upload ---
title_col, upload_col = st.columns([2, 1])
with title_col:
    st.title("RAG Chatbot")
    st.caption(
        "Upload a PDF or text file and ask questions about its content. "
        "using Retrieval-Augmented Generation (RAG)."
    )

with upload_col:
    uploaded = st.file_uploader(
        "📄 Upload PDF or Text File", type=["pdf", "txt"], label_visibility="collapsed"
    )
    if uploaded:
        st.session_state.uploaded_file = uploaded

# --- Session state ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_logs()


# --- Display previous chat history ---
for timestamp in sorted(st.session_state.chat_history.keys()):
    msg = st.session_state.chat_history[timestamp]
    with st.chat_message("user"):
        st.markdown(msg["query"])
    with st.chat_message("assistant"):
        st.markdown(msg["result"])

# --- Handle chat input ---
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.uploaded_file:
        st.warning("📂 Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving answer..."):
                try:
                    response = retriever_qa(st.session_state.uploaded_file, prompt)
                    result = response.get("result", "No result found.")
                    st.markdown(result)

                    # Save to chat_history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history[timestamp] = {
                        "query": prompt,
                        "result": result,
                    }
                    save_chat_logs(st.session_state.chat_history)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# --- Display sources ---
if prompt and st.session_state.uploaded_file:
    response = (
        response
        if "response" in locals()
        else retriever_qa(st.session_state.uploaded_file, prompt)
    )
    source_docs = response.get("source_documents", [])

    with st.expander("📚 Source Documents"):
        if source_docs:
            for i, doc in enumerate(source_docs, 1):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                with st.expander(f"Source {i}", expanded=False):
                    st.markdown(f"**Content:**\n{content}")
                    if metadata:
                        st.markdown(f"**Metadata:**\n`{metadata}`")
        else:
            st.info("No source documents found.")

# delete chat_logs file if it exists also refresh the chat history
if st.button("Clear Chat History"):
    if os.path.exists(CHAT_LOG_FILE):
        os.remove(CHAT_LOG_FILE)
    st.session_state.chat_history = {}
    st.session_state.uploaded_file = None  # Clear uploaded file
    save_chat_logs(st.session_state.chat_history)
    st.success("Chat history cleared.")
    # Show success message before rerun to avoid RuntimeError
    st.rerun()  # Refresh the page to clear chat messages
