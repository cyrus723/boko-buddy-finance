import os
from datetime import datetime

import streamlit as st
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI


def get_openai_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def has_local_index():
    return os.path.isdir("./storage") and os.path.exists("./storage/index_store.json")


if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(page_title="TXST Dr. Yi 3313 Tutor", layout="wide")
st.title("Finance 3313 Financial Management Tutor")
st.write("1. Explain the term, initial public offering.")
st.write("2. Ask some multiple choice questions on net present value.")
st.write("3. Ask short essay questions on capital structure.")
st.write("4. Ask some word problems on stock valuation.")

with st.sidebar:
    st.header("Tutor Settings")
    mode = st.radio(
        "Tutoring Style",
        ["Strict (Lecture Only)", "Supportive (Lecture + Textbook)", "General AI (No RAG)"],
    )
    temp = st.slider("Creativity/Temperature", 0.0, 1.0, 0.1)
    is_summary_mode = st.toggle("Deep Search / Summary Mode")

    if st.button("Clear Chat History"):
        st.session_state.messages = []

with st.sidebar:
    st.markdown("---")
    st.header("Study Tools")

    if st.session_state.messages:
        chat_export = "# Study Guide\n\n"
        for msg in st.session_state.messages:
            role = "Student" if msg["role"] == "user" else "Tutor"
            chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"

        st.download_button(
            label="Download Study Guide (.md)",
            data=chat_export,
            file_name="study_notes.md",
            mime="text/markdown",
            help="Click to save this conversation as a study guide.",
        )
    else:
        st.caption("Chat with the tutor to generate a study guide.")


@st.cache_resource(show_spinner=False)
def get_query_engine(selected_mode, temperature, summary_mode):
    api_key = get_openai_api_key()
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    if selected_mode == "General AI (No RAG)":
        from llama_index.core.chat_engine import SimpleChatEngine

        return SimpleChatEngine.from_defaults(
            llm=OpenAI(model="gpt-4o", temperature=temperature, api_key=api_key),
            system_prompt=(
                f"You are a helpful AI assistant. Today is {current_date}. "
                "Answer using only your general knowledge."
            ),
        )

    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    if selected_mode == "Strict (Lecture Only)":
        system_text = (
            f"You are an expert Teaching Assistant for the Financial Management course at TXST for Spring 2026. Today is {current_date}. "
            "Use ONLY the 'lecture' source files to answer. "
            "If the answer is not in the slides, refer the student to the textbook."
        )
    else:
        system_text = (
            f"You are an expert Teaching Assistant for the Financial Management course at TXST for Spring 2026. Today is {current_date}. "
            "Your tone is professional, encouraging, and technically precise. "
            "1. Use the 'lecture' sources for high-level concepts and 'textbook' for deep technical details. "
            "2. If code is involved, Python examples where appropriate. "
            "3. Always cite your sources at the end of your explanation. "
            "4. If you do not know the answer based on the provided context, offer to help the student formulate a question for office hours."
        )

    Settings.llm = OpenAI(model="gpt-4o", temperature=temperature, api_key=api_key)

    similarity_top_k = 15 if summary_mode else 5
    return index.as_chat_engine(
        chat_mode="context",
        system_prompt=system_text,
        similarity_top_k=similarity_top_k,
        streaming=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


api_key = get_openai_api_key()
index_available = has_local_index()

if not api_key:
    st.error("OpenAI API key not found. Add `OPENAI_API_KEY` in Streamlit secrets or your environment before running the app.")
    st.info('For Streamlit Community Cloud, open App Settings -> Secrets and add `OPENAI_API_KEY = "..."`.')

if not index_available:
    st.warning("RAG index not found in `./storage`. The standard GPT column can still run, but the tutor column needs the generated index files.")


if prompt := st.chat_input("Financial Management Question...", disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    col_rag, col_general = st.columns(2)
    rag_response = None

    with col_rag:
        st.subheader("TXST AI Tutor")
        with st.chat_message("assistant"):
            if index_available:
                rag_engine = get_query_engine("Supportive (Lecture + Textbook)", temp, is_summary_mode)
                with st.spinner("Consulting course materials..."):
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    enhanced_prompt = f"(Context: Today is {today_str}). User asks: {prompt}"
                    rag_response = rag_engine.chat(enhanced_prompt)
                    st.markdown(rag_response.response)

                    if hasattr(rag_response, "source_nodes") and rag_response.source_nodes:
                        with st.expander("Sources Found"):
                            for node in rag_response.source_nodes[:3]:
                                st.caption(
                                    f"From: {node.metadata.get('source_file')} "
                                    f"(Page {node.metadata.get('page_label')})"
                                )
            else:
                st.info("RAG response unavailable because `./storage` is not present in this deployment.")

    with col_general:
        st.subheader("Standard GPT")
        with st.chat_message("assistant"):
            gen_engine = get_query_engine("General AI (No RAG)", temp, is_summary_mode)
            with st.spinner("Thinking generally..."):
                today_str = datetime.now().strftime("%Y-%m-%d")
                enhanced_prompt = f"(Context: Today is {today_str}). User asks: {prompt}"
                gen_response = gen_engine.chat(enhanced_prompt)
                st.markdown(gen_response.response)

    if rag_response is not None:
        st.session_state.messages.append({"role": "assistant", "content": rag_response.response})
