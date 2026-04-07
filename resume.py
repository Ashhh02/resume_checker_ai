import os
import tempfile
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an expert resume reviewer and career advisor with 15 years of HR and recruiting experience.

You are given chunks of the user's resume as context. Your job is to:
- Analyze the resume content thoroughly
- Give specific, actionable feedback (not generic advice)
- Suggest concrete rewrites and improvements with examples
- Recommend suitable career paths based on the skills and experience you see
- Point out both strengths and weaknesses per section

RULES:
- Only base your answer on the provided resume context
- Never make up job titles, companies, or skills not present in the context
- If the context does not contain enough information to answer, say: "I couldn't find that information in your resume. Could you clarify or add more detail?"
- Format responses clearly: use bullet points, bold headers, and examples
- Be encouraging but honest
"""

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_vector_collection():
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="nomic-embed-text",
        timeout=60,
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
    )

    return splitter.split_documents(docs)

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()

    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

def query_collection(prompt: str, n_results: int = 5):
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def re_rank_cross_encoders(prompt: str, documents: list[str]):
    encoder = load_cross_encoder()

    ranks = encoder.rank(prompt, documents, top_k=3)

    relevant_text = ""
    relevant_ids = []

    for r in ranks:
        relevant_text += documents[r["corpus_id"]] + "\n\n"
        relevant_ids.append(r["corpus_id"])

    return relevant_text, relevant_ids

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{prompt}",
            },
        ],
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]

st.set_page_config(
    page_title="Resume & Career Advisor",
    page_icon="💼",
    layout="wide"
)
    
with st.sidebar:
    st.title("💼 Resume Explorer")
    st.divider()

    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        accept_multiple_files=False,
        help="Your resume is processed locally. Nothing is sent to the cloud."
    )

    if st.button("⚡ Process Resume", use_container_width=True) and uploaded_file:
        with st.spinner("Reading your resume..."):
            try:
                name = uploaded_file.name.replace(" ", "_")
                splits = process_document(uploaded_file)
                add_to_vector_collection(splits, name)
                st.success("✅ Resume loaded! Ask me anything below.")
                st.session_state["resume_loaded"] = True
                st.session_state["resume_name"] = uploaded_file.name
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if st.session_state.get("resume_loaded"):
        st.info(f"📎 Active: {st.session_state.get('resume_name')}")

    st.divider()

    st.subheader("💡 Quick Questions")
    quick_questions = [
        "Is my resume strong overall?",
        "What are the weaknesses in my resume?",
        "What jobs or roles fit my experience?",
        "Rewrite my professional summary section.",
        "How can I improve my skills section?",
        "What salary range should I expect?",
        "Give me 5 interview questions based on my resume.",
        "What is missing from my resume?",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state["quick_input"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Upload a new resume or ask a new question!"
            }
        ]
        st.session_state["resume_loaded"] = False
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your Career Advisor. Upload your resume in the sidebar, then ask me anything — I'll review it section by section, suggest improvements, and help you find the right career path. 💼"
        }
    ]

st.title("💼 Resume & Career Advisor")
st.caption("Upload your resume → ask questions → get expert AI feedback")
st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Handle quick question injection
default_input = st.session_state.pop("quick_input", "")

# Chat input
prompt = st.chat_input(
    "Ask about your resume or career...",
    key="chat_input"
) or (default_input if default_input else None)

if prompt:
    # Guard: warn if no resume loaded
    if not st.session_state.get("resume_loaded"):
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Please upload and process your resume first using the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Analyzing your resume...", expanded=False) as status:
            try:
                status.write("🔍 Searching resume content...")
                results = query_collection(prompt)

                if not results.get("documents") or len(results["documents"][0]) == 0:
                    status.update(label="❌ Nothing found", state="error")
                    st.warning("Could not find relevant content. Try rephrasing your question.")
                    st.stop()

                docs = results["documents"][0]

                status.write("📊 Re-ranking relevant sections...")
                context, relevant_ids = re_rank_cross_encoders(prompt, docs)

                status.write("✍️ Generating advice...")
                status.update(label="✅ Done!", state="complete")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")
                st.stop()

        # Stream the response
        response_placeholder = st.empty()
        full_response = ""
        for chunk in call_llm(context, prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        # Collapsible debug info
        with st.expander("🔎 Resume sections used for this answer"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:** {doc[:300]}...")