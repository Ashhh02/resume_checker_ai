import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error
except ImportError:  # pragma: no cover - app can still run without MySQL support
    mysql_connector = None
    Error = Exception

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