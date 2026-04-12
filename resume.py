import os
import tempfile
import uuid

import ollama
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error
except ImportError:  # pragma: no cover - app can still run without MySQL support
    mysql_connector = None
    Error = Exception


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


def re_rank_cross_encoders(prompt: str, documents: list[str]):
    encoder = load_cross_encoder()
    ranks = encoder.rank(prompt, documents, top_k=3)

    relevant_text = ""
    relevant_ids = []

    for r in ranks:
        relevant_text += documents[r["corpus_id"]] + "\n\n"
        relevant_ids.append(r["corpus_id"])

    return relevant_text, relevant_ids


def search_resume_context(prompt: str, session_id: str, file_name: str | None = None):
    docs = load_resume_chunks(session_id, file_name)
    if not docs:
        docs = load_resume_chunks_for_latest_upload()
        if not docs:
            return "", [], []

    context, relevant_ids = re_rank_cross_encoders(prompt, docs)
    return context, relevant_ids, docs


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


st.title("Resume & Career Advisor")
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
                context, relevant_ids, docs = search_resume_context(
                    prompt,
                    st.session_state["session_id"],
                    st.session_state.get("resume_name"),
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

        response_placeholder = st.empty()
        full_response = ""
        for chunk in call_llm(context, prompt):
            full_response += chunk
            response_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
        save_chat_message(st.session_state["session_id"], "assistant", full_response)

        with st.expander("Resume sections used for this answer"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i + 1}:** {doc[:300]}...")
