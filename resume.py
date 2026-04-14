import os
import tempfile
import uuid

import ollama
import streamlit as st
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from bs4 import BeautifulSoup

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error
except ImportError:  # pragma: no cover - app can still run without MySQL support
    mysql_connector = None
    Error = Exception


system_prompt = """
You are a senior resume reviewer, recruiter, and career advisor.

You analyze only the provided resume context and optional job description.

Your responsibilities:
- Score the resume from 0 to 100
- If a job description is provided, calculate a match score from 0 to 100
- Identify missing skills, role gaps, and weak sections
- Rewrite resume content when asked
- Suggest career paths, next-step roles, and improvement priorities
- Give specific, actionable feedback with examples

Response rules:
- Use only facts found in the provided context
- Do not invent employers, degrees, certifications, tools, or achievements
- If information is missing, say so clearly
- Avoid generic advice like "tailor your resume" unless you explain exactly how
- Keep the response structured with these exact sections when possible:
  1. **Score**
  2. **Match Analysis**
  3. **Missing Skills**
  4. **Resume Rewrite**
  5. **Career Recommendations**
  6. **Next Actions**
- Use bullets, short paragraphs, and concise examples
- If rewriting content, provide improved text in quotation marks or a code block
- Be direct, helpful, and honest
"""


def get_mysql_config() -> dict[str, object]:
    return {
        "host": os.getenv("DB_HOST") or "127.0.0.1",
        "port": int(os.getenv("DB_PORT") or "3306"),
        "user": os.getenv("DB_USER") or "root",
        "password": os.getenv("DB_PASSWORD") or "4Shyn.zyyy20",
        "database": os.getenv("DB_NAME") or "resume_ai",
    }


def connect_mysql():
    if mysql_connector is None:
        st.session_state["mysql_error"] = "mysql-connector-python is not installed"
        return None

    try:
        return mysql_connector.connect(**get_mysql_config())
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return None


@st.cache_resource
def test_mysql_connection() -> bool:
    connection = connect_mysql()
    if connection is None:
        return False

    try:
        return connection.is_connected()
    finally:
        connection.close()


def get_or_create_user_id(session_id: str) -> int | None:
    connection = connect_mysql()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            """
            INSERT INTO users (session_id)
            VALUES (%s)
            """,
            (session_id,),
        )
        connection.commit()
        return cursor.lastrowid
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return None
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def save_resume_upload(session_id: str, file_name: str, chunk_count: int) -> int | None:
    connection = connect_mysql()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        user_id = get_or_create_user_id(session_id)
        if user_id is None:
            raise RuntimeError("Could not create or load user row.")

        cursor.execute(
            """
            INSERT INTO resumes (user_id, file_name, chunk_count)
            VALUES (%s, %s, %s)
            """,
            (user_id, file_name, chunk_count),
        )
        connection.commit()
        return cursor.lastrowid
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return None
    except RuntimeError as exc:
        st.session_state["mysql_error"] = str(exc)
        return None
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def save_resume_chunks(resume_id: int, chunks: list[Document]) -> None:
    connection = connect_mysql()
    if connection is None:
        return

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.executemany(
            """
            INSERT INTO resume_chunks (resume_id, chunk_index, chunk_text)
            VALUES (%s, %s, %s)
            """,
            [
                (resume_id, idx, chunk.page_content)
                for idx, chunk in enumerate(chunks)
            ],
        )
        connection.commit()
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def load_resume_chunks(session_id: str, file_name: str | None = None) -> list[str]:
    connection = connect_mysql()
    if connection is None:
        return []

    cursor = None
    try:
        cursor = connection.cursor()
        if file_name:
            cursor.execute(
                """
                SELECT c.chunk_text
                FROM users u
                JOIN resumes r ON r.user_id = u.id
                JOIN resume_chunks c ON c.resume_id = r.id
                WHERE u.session_id = %s AND r.file_name = %s
                ORDER BY c.chunk_index ASC, c.id ASC
                """,
                (session_id, file_name),
            )
        else:
            cursor.execute(
                """
                SELECT c.chunk_text
                FROM users u
                JOIN resumes r ON r.user_id = u.id
                JOIN resume_chunks c ON c.resume_id = r.id
                WHERE u.session_id = %s
                ORDER BY r.uploaded_at DESC, r.id DESC, c.chunk_index ASC, c.id ASC
                """,
                (session_id,),
            )
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return []
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def load_latest_resume_name() -> str | None:
    connection = connect_mysql()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT file_name
            FROM resumes
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return None
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def load_latest_resume_reference() -> tuple[int, str] | None:
    connection = connect_mysql()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT r.id, r.file_name
            FROM resumes r
            ORDER BY r.uploaded_at DESC, r.id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row[0], row[1]
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return None
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def load_resume_chunks_for_latest_upload() -> list[str]:
    latest_ref = load_latest_resume_reference()
    if latest_ref is None:
        return []

    latest_resume_id, _ = latest_ref
    connection = connect_mysql()
    if connection is None:
        return []

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT chunk_text
            FROM resume_chunks
            WHERE resume_id = %s
            ORDER BY chunk_index ASC, id ASC
            """,
            (latest_resume_id,),
        )
        return [row[0] for row in cursor.fetchall()]
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
        return []
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def save_chat_message(session_id: str, role: str, content: str) -> None:
    connection = connect_mysql()
    if connection is None:
        return

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO chat_messages (session_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (session_id, role, content),
        )
        connection.commit()
    except Error as exc:
        st.session_state["mysql_error"] = str(exc)
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def extract_job_description(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("Please paste a job URL first.")

    response = requests.get(
        url.strip(),
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    parts = []
    if soup.title and soup.title.get_text(strip=True):
        parts.append(f"Title: {soup.title.get_text(strip=True)}")

    for meta_name in ("description", "og:description"):
        meta = soup.find("meta", attrs={"name": meta_name}) or soup.find(
            "meta", attrs={"property": meta_name}
        )
        if meta and meta.get("content"):
            parts.append(f"{meta_name.title()}: {meta['content'].strip()}")

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    parts.append(text)

    job_description = "\n\n".join(parts)
    return job_description[:3000]


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    if uploaded_file is None:
        raise ValueError("Please upload a PDF resume first.")
    if getattr(uploaded_file, "size", 0) == 0:
        raise ValueError("The uploaded PDF is empty.")

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

    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    return chunks


def dedupe_documents(documents: list[str]) -> list[str]:
    seen = set()
    unique_documents = []
    for doc in documents:
        normalized = " ".join(doc.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_documents.append(doc)
    return unique_documents


def re_rank_cross_encoders(prompt: str, documents: list[str], top_k: int = 4):
    encoder = load_cross_encoder()
    unique_documents = dedupe_documents(documents)
    if not unique_documents:
        return "", [], []

    top_k = min(max(3, top_k), 5, len(unique_documents))
    ranks = encoder.rank(prompt, unique_documents, top_k=top_k)

    relevant_text = []
    relevant_ids = []
    seen_ids = set()
    for r in ranks:
        corpus_id = r["corpus_id"]
        if corpus_id in seen_ids:
            continue
        seen_ids.add(corpus_id)
        relevant_text.append(unique_documents[corpus_id])
        relevant_ids.append(corpus_id)

    return "\n\n".join(relevant_text), relevant_ids, relevant_text


def search_resume_context(
    prompt: str,
    session_id: str,
    file_name: str | None = None,
    job_description: str | None = None,
):
    docs = load_resume_chunks(session_id, file_name)
    if not docs:
        docs = load_resume_chunks_for_latest_upload()
        if not docs:
            return "", [], []

    search_query = prompt
    if job_description:
        search_query = f"{prompt}\n\nJob Description:\n{job_description}"

    context, relevant_ids, selected_chunks = re_rank_cross_encoders(search_query, docs)
    return context, relevant_ids, selected_chunks


def extract_score(text: str) -> int | None:
    import re

    patterns = [
        r"(?:Match Score|Resume Score|Score)\s*[:\-]?\s*(\d{1,3})\s*/\s*100",
        r"(?:Match Score|Resume Score|Score)\s*[:\-]?\s*(\d{1,3})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(score, 100))
    return None


def call_llm(context: str, prompt: str, job_description: str | None = None):
    user_content = (
        "Use the resume context below to answer the request.\n\n"
        f"Resume Context:\n{context}\n\n"
    )
    if job_description and job_description.strip():
        user_content += f"Job Description:\n{job_description.strip()}\n\n"
    user_content += f"User Request:\n{prompt}"

    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        options={
            "temperature": 0.2,
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]


st.set_page_config(
    page_title="Resume Analysis and Career Recommendation System",
    page_icon="CV",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

with st.sidebar:
    st.title("Resume Explorer")
    st.divider()

    if test_mysql_connection():
        st.success("MySQL connected")
    else:
        mysql_error = st.session_state.get("mysql_error")
        if mysql_error:
            st.warning(f"MySQL unavailable: {mysql_error}")
        else:
            st.info("MySQL is not connected yet.")

    st.subheader("Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        accept_multiple_files=False,
        help="Your resume is processed locally. Nothing is sent to the cloud.",
    )

    st.subheader("Job Matching")
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
                splits = process_document(uploaded_file)
                resume_id = save_resume_upload(
                    st.session_state["session_id"],
                    uploaded_file.name,
                    len(splits),
                )
                if resume_id is None:
                    raise RuntimeError("Could not save resume metadata to MySQL.")
                save_resume_chunks(resume_id, splits)
                st.success("Resume loaded. Ask me anything below.")
                st.session_state["resume_loaded"] = True
                st.session_state["resume_name"] = uploaded_file.name
            except Exception as exc:
                st.error(f"Error processing file: {exc}")

    if st.session_state.get("resume_loaded"):
        st.info(f"Active: {st.session_state.get('resume_name')}")

    st.divider()

    st.subheader("Quick Questions")
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
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Upload a new resume or ask a new question!",
            }
        ]
        st.session_state["resume_loaded"] = False
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your Career Advisor. Upload your resume in the sidebar, then "
                "ask me anything and I'll review it section by section, suggest "
                "improvements, and help you find the right career path."
            ),
        }
    ]


st.title("Resume Analysis and Career Recommendation System")
st.caption("Upload your resume -> ask questions -> get expert AI feedback")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"], avatar="assistant" if msg["role"] == "assistant" else "user"
    ):
        st.markdown(msg["content"])

default_input = st.session_state.pop("quick_input", "")

prompt = st.chat_input(
    "Ask about your resume or career...",
    key="chat_input",
) or (default_input if default_input else None)

if prompt:
    if not st.session_state.get("resume_loaded"):
        with st.chat_message("assistant", avatar="assistant"):
            st.warning("Please upload and process your resume first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state["session_id"], "user", prompt)

    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="assistant"):
        with st.status("Analyzing your resume...", expanded=False) as status:
            try:
                status.write("Searching resume content...")
                context, relevant_ids, selected_chunks = search_resume_context(
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
                status.update(label="Done", state="complete")
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
            st.markdown("**Match Score**")
            st.progress(match_score / 100.0)
            st.caption(f"{match_score}/100")

        with st.expander("Resume sections used for this answer"):
            for i, chunk in enumerate(selected_chunks):
                preview = chunk[:300].replace("\n", " ")
                st.markdown(f"- **Chunk {i + 1}**: {preview}...")
