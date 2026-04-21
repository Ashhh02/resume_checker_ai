import uuid
from pathlib import Path

import streamlit as st

from resume_ai import (
    DEFAULT_MODEL,
    call_llm,
    extract_job_description,
    extract_score,
    search_resume_context,
)
from resume_db import (
    load_chat_messages,
    process_document,
    save_chat_message,
    save_resume_upload,
    save_resume_text,
    test_mysql_connection,
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(46, 134, 222, 0.12), transparent 24%),
                    radial-gradient(circle at top right, rgba(9, 132, 227, 0.10), transparent 18%),
                    linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
                color: #e5eef9;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
                max-width: 1280px;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.98));
                border-right: 1px solid rgba(148, 163, 184, 0.12);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 1.25rem;
            }

            div[data-testid="stHeader"] {
                background: transparent;
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.80));
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 24px;
                padding: 1.6rem 1.7rem;
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
            }

            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: #93c5fd;
                font-size: 0.72rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }

            .hero-title {
                font-size: 2.05rem;
                font-weight: 800;
                line-height: 1.05;
                color: #f8fafc;
                margin-bottom: 0.45rem;
            }

            .hero-subtitle {
                color: rgba(226, 232, 240, 0.78);
                font-size: 0.98rem;
                margin-bottom: 1rem;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.15);
                color: #dbeafe;
                font-size: 0.82rem;
            }

            .section-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.15em;
                color: #93c5fd;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }

            .soft-note {
                color: rgba(226, 232, 240, 0.7);
                font-size: 0.92rem;
                margin-bottom: 0.5rem;
            }

            .stTextInput input, .stTextArea textarea {
                background: rgba(15, 23, 42, 0.88) !important;
                color: #f8fafc !important;
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                border-radius: 14px !important;
            }

            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder {
                color: rgba(203, 213, 225, 0.55) !important;
            }

            .stButton > button {
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: linear-gradient(135deg, #334155, #1f2937);
                color: #f8fafc;
                font-weight: 600;
                padding: 0.65rem 1rem;
            }

            .stButton > button:hover {
                border-color: rgba(96, 165, 250, 0.55);
                transform: translateY(-1px);
            }

            div[data-testid="stChatMessage"] {
                border-radius: 18px;
                padding-top: 0.25rem;
                padding-bottom: 0.25rem;
            }

            [data-testid="stChatMessageAvatarUser"],
            [data-testid="stChatMessageAvatarAssistant"] {
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    resume_loaded = bool(st.session_state.get("resume_loaded"))
    job_loaded = bool((st.session_state.get("job_description") or "").strip())
    job_url = bool((st.session_state.get("job_url") or "").strip())

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">AI Resume Analyzer + Job Matching</div>
            <div class="hero-title">Resume Analysis and Career Recommendation System</div>
            <div class="hero-subtitle">
                Upload a resume, match it against a job description or URL, and get a structured score, skill gaps, and practical next steps.
            </div>
            <div class="pill-row">
                <span class="pill">Local Ollama</span>
                <span class="pill">MySQL</span>
                <span class="pill">Direct resume context</span>
                <span class="pill">{'Resume loaded' if resume_loaded else 'Resume not loaded'}</span>
                <span class="pill">{'Job source ready' if (job_loaded or job_url) else 'No job source yet'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip() -> None:
    resume_loaded = bool(st.session_state.get("resume_loaded"))
    resume_name = st.session_state.get("resume_name") or "No resume loaded"
    job_text = (st.session_state.get("job_description") or "").strip()
    selected_model = st.session_state.get("ollama_model", DEFAULT_MODEL)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resume", "Ready" if resume_loaded else "Pending", resume_name)
    with col2:
        st.metric("Job Source", "Loaded" if job_text else "Pending", "URL or pasted text")
    with col3:
        st.metric("Analysis", "Direct Context", selected_model)


st.set_page_config(
    page_title="Resume Analysis and Career Recommendation System",
    page_icon="CV",
    layout="wide",
)

inject_styles()
USER_AVATAR_PATH = Path(__file__).parent / "assets" / "user.png"
ASSISTANT_AVATAR_PATH = Path(__file__).parent / "assets" / "chatbot.png"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = DEFAULT_MODEL

if "messages" not in st.session_state:
    db_messages = load_chat_messages(st.session_state["session_id"])
    if db_messages:
        st.session_state.messages = db_messages
    else:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Upload your resume, optionally add a job description or job URL, and I'll give you a structured match analysis with score, gaps, and suggestions."
                ),
            }
        ]

with st.sidebar:
    st.markdown('<div class="section-label">Nash Final Project</div>', unsafe_allow_html=True)
    st.markdown("### Resume Explorer")
    st.markdown(
        '<div class="soft-note">Upload a resume, load a job description, then ask for a fit analysis.</div>',
        unsafe_allow_html=True,
    )

    if test_mysql_connection():
        st.success("MySQL connected")
    else:
        mysql_error = st.session_state.get("mysql_error")
        if mysql_error:
            st.warning(f"MySQL unavailable: {mysql_error}")
        else:
            st.info("MySQL is not connected yet.")

    st.markdown('<div class="section-label">Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        accept_multiple_files=False,
        help="Your resume is processed locally. Nothing is sent to the cloud.",
    )

    st.markdown('<div class="section-label">AI Model</div>', unsafe_allow_html=True)
    recommended_models = [
        "qwen2.5:7b-instruct",
        "nomic-embed-text:latest",
        "llama3.2:3b",
    ]
    default_model_index = (
        recommended_models.index(st.session_state["ollama_model"])
        if st.session_state["ollama_model"] in recommended_models
        else 0
    )
    st.session_state["ollama_model"] = st.selectbox(
        "Choose Ollama model",
        options=recommended_models,
        index=default_model_index,
        help=(
            "Recommended: qwen2.5:7b-instruct for better resume matching accuracy. "
            "Make sure the model is installed in Ollama first."
        ),
    )

    st.markdown('<div class="section-label">Job Match</div>', unsafe_allow_html=True)
    st.session_state["job_url"] = st.text_input(
        "Paste Job URL",
        value=st.session_state.get("job_url", ""),
        placeholder="https://... (LinkedIn, JobStreet, company careers page, etc.)",
    )
    if st.button("Fetch Job Description", use_container_width=True):
        try:
            with st.spinner("Extracting job description from URL..."):
                extracted = extract_job_description(st.session_state.get("job_url", ""))
                st.session_state["job_description"] = extracted
                st.success("Job description loaded from URL.")
        except Exception as exc:
            st.error(f"Could not extract job description: {exc}")

    st.session_state["job_description"] = st.text_area(
        "Paste Job Description",
        value=st.session_state.get("job_description", ""),
        height=180,
        placeholder="Paste a job description here, or fetch one from a job URL above.",
    )

    if st.button("Process Resume", use_container_width=True) and uploaded_file:
        with st.spinner("Reading your resume..."):
            try:
                resume_text = process_document(uploaded_file)
                resume_id = save_resume_upload(
                    st.session_state["session_id"],
                    uploaded_file.name,
                )
                if resume_id is None:
                    mysql_error = st.session_state.get("mysql_error") or "Unknown MySQL error."
                    raise RuntimeError(f"Could not save resume metadata to MySQL. {mysql_error}")
                save_resume_text(resume_id, resume_text)
                st.success("Resume loaded. Ask me anything below.")
                st.session_state["resume_loaded"] = True
                st.session_state["resume_name"] = uploaded_file.name
            except Exception as exc:
                st.error(f"Error processing file: {exc}")

    if st.session_state.get("resume_loaded"):
        st.info(f"Active: {st.session_state.get('resume_name')}")

    st.divider()

    with st.expander("Quick Prompts", expanded=False):
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

        for i in range(0, len(quick_questions), 2):
            left = quick_questions[i]
            right = quick_questions[i + 1] if i + 1 < len(quick_questions) else None
            cols = st.columns(2)
            with cols[0]:
                if st.button(left, use_container_width=True, key=f"quick_{left}"):
                    st.session_state["quick_input"] = left
            with cols[1]:
                if right and st.button(right, use_container_width=True, key=f"quick_{right}"):
                    st.session_state["quick_input"] = right

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["session_id"] = uuid.uuid4().hex
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Upload a new resume or ask a new question!",
            }
        ]
        st.session_state["resume_loaded"] = False
        st.rerun()

render_hero()
st.write("")
render_status_strip()
st.write("")

st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="soft-note">Ask for a score, skill gap analysis, rewrite suggestions, or job-fit feedback.</div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    if msg["role"] == "user" and USER_AVATAR_PATH.exists():
        with st.chat_message(msg["role"], avatar=str(USER_AVATAR_PATH)):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and ASSISTANT_AVATAR_PATH.exists():
        with st.chat_message(msg["role"], avatar=str(ASSISTANT_AVATAR_PATH)):
            st.markdown(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

default_input = st.session_state.pop("quick_input", "")

prompt = st.chat_input(
    "Ask about your resume or career...",
    key="chat_input",
) or (default_input if default_input else None)

if prompt:
    if not st.session_state.get("resume_loaded"):
        if ASSISTANT_AVATAR_PATH.exists():
            with st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR_PATH)):
                st.warning("Please upload and process your resume first using the sidebar.")
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process your resume first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state["session_id"], "user", prompt)

    if USER_AVATAR_PATH.exists():
        with st.chat_message("user", avatar=str(USER_AVATAR_PATH)):
            st.markdown(prompt)
    else:
        with st.chat_message("user"):
            st.markdown(prompt)

    if ASSISTANT_AVATAR_PATH.exists():
        assistant_chat = st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR_PATH))
    else:
        assistant_chat = st.chat_message("assistant")

    with assistant_chat:
        with st.status("Analyzing your resume...", expanded=False) as status:
            try:
                status.write("Searching resume content...")
                context, selected_sections = search_resume_context(
                    prompt,
                    st.session_state["session_id"],
                    st.session_state.get("resume_name"),
                    st.session_state.get("job_description"),
                )

                if not context.strip():
                    status.update(label="Nothing found", state="error")
                    st.warning(
                        "Could not find relevant content. Try processing your resume again."
                    )
                    st.stop()

                status.write("Generating advice...")
                status.update(label="Done Analyzing of Nash AI", state="complete")
            except Exception as exc:
                status.update(label="Error", state="error")
                st.error(f"Error: {exc}")
                st.stop()

        with st.spinner("Generating structured answer..."):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in call_llm(
                context,
                prompt,
                st.session_state.get("job_description"),
                st.session_state.get("ollama_model", DEFAULT_MODEL),
            ):
                full_response += chunk
                response_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
        save_chat_message(st.session_state["session_id"], "assistant", full_response)

        match_score = extract_score(full_response)
        if match_score is not None:
            st.markdown("#### Match Score")
            st.progress(match_score / 100.0)
            st.caption(f"{match_score}/100")

        with st.expander("Resume context used for this answer"):
            st.caption(
                "Only the most relevant resume snippets for this question are shown here."
            )
            for index, section in enumerate(selected_sections, start=1):
                cleaned_section = section.strip()
                preview = cleaned_section[:500]
                suffix = "..." if len(cleaned_section) > 500 else ""
                st.markdown(f"**Snippet {index}**")
                st.caption(preview + suffix)