"""
interface.py
------------
Streamlit web interface for the Graduate Programme Python Assessment project.
- Provides chat and document Q&A modes.
- Handles user input, displays responses, and manages session state.
- Ensures all structured data is displayed as a DataFrame for easy analysis.
"""

import streamlit as st
from agent import GraphAgent
import time
from uuid import uuid4
from logger import setup_logger
from retriever import Retriever
import json

def close_chat():
    """
    Ends the chat session if there is inactivity for more than 4 minutes (240 seconds).
    """
    st.session_state.logger.info('Checking for chat inactivity.')
    current_time = time.time()
    time_taken = current_time - st.session_state.time_taken
    if time_taken > 240:
        st.session_state.logger.info('Chat ended due to inactivity. Time Elapsed: %d seconds', int(time_taken))
        with st.chat_message("system"):
            st.markdown(f"**Chat ended due to inactivity. Time Elapsed: {int(time_taken)} seconds**")
            st.stop()

def page_switch():
    """
    Resets session state for a new chat or document Q&A session.
    """
    st.session_state.messages = []
    st.session_state.config = {"configurable": {"thread_id": f"{uuid4()}"}}
    st.session_state.doc_name = None

def clean_response(response):
    """
    Utility to display the response as a DataFrame if possible, or as text otherwise.
    Handles stringified JSON arrays, lists of dicts, and dicts.
    """
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                st.dataframe(parsed)
            else:
                st.write(response)
        except Exception:
            st.write(response)
    elif isinstance(response, list) and all(isinstance(item, dict) for item in response):
        st.dataframe(response)
    elif isinstance(response, dict):
        st.dataframe([response])
    else:
        st.write(response)

def chat_interface():
    """
    Streamlit page for PD Chat (timesheet Q&A).
    Handles user input, displays chat history, and shows results as DataFrames.
    """
    page_switch()
    st.title("💬 PD Chat")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    user_input = st.chat_input("Ask something...", on_submit=close_chat)
    # Handle input
    if user_input:
        st.session_state.time_taken = time.time()
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Cooking up a response..."):
            for chunk in st.session_state.chatbot.stream({"question": user_input, "session_id": st.session_state.config["configurable"]["thread_id"], "prompt_type": "timesheet prompt"}, st.session_state.config, stream_mode=["custom", "values"]):
                if chunk[0] == "custom":
                    st.write(chunk[1])
                else:
                    response = chunk[1].get("answer", "")
        with st.chat_message("assistant"):
            clean_response(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            page_switch()

def document_qa():
    """
    Streamlit page for Document Q&A.
    Allows document upload, question input, and displays answers/metadata.
    """
    page_switch()
    st.title("📄 Document Q&A")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"], accept_multiple_files=False)
    if uploaded_file:
        
        with st.spinner("Processing document..."):
            st.session_state.retriever.save_doc(uploaded_file)

        st.success("File uploaded!")
        
        if st.session_state.doc_name is None:
            st.session_state.doc_name = uploaded_file.name
        query = st.text_input("Ask a question about the document:", on_change=close_chat)
        if query:
            st.session_state.time_taken = time.time()
            st.write(f"Answering question based on {uploaded_file.name}:")
            with st.spinner("Cooking up a response..."):
                for chunk in st.session_state.chatbot.stream({"question": query, "session_id": st.session_state.config["configurable"]["thread_id"], "prompt_type": "document prompt", "doc_name": st.session_state.doc_name}, st.session_state.config, stream_mode=["custom", "values"]):
                    if chunk[0] == "custom":
                        st.write(chunk[1])
                    else:
                        response = chunk[1].get("answer", "")
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select a page to navigate:")

# Default to chat_interface on first load
if "page" not in st.session_state:
    st.session_state.page = "pd_chat"

if "logger" not in st.session_state:
    st.session_state.logger = setup_logger(__name__, 'interface.log')

if "chatbot" not in st.session_state:
    st.session_state.chatbot = GraphAgent().get_graph()
    st.session_state.logger.info('Chatbot graph initialized.')

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()
    st.session_state.logger.info('Retriever initialized.')

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": f"{uuid4()}"}}
    st.session_state.logger.info('Session config initialized.')

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.logger.info('Session messages initialized.')

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "time_taken" not in st.session_state:
    st.session_state.time_taken = time.time()   
    st.session_state.logger.info('Session time_taken initialized.')

if st.sidebar.button("PD Chat", key="pd_chat"):
    st.session_state.page = "pd_chat"
if st.sidebar.button("Document Q&A", key="document_qa"):
    st.session_state.page = "document_qa"

# Render the selected page
if st.session_state.page == "pd_chat":
    chat_interface()
elif st.session_state.page == "document_qa":
    document_qa()
