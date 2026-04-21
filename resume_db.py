import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv

# In Streamlit, root-level secrets may already be injected into environment
# variables. For local development, prefer values from .env so the app and the
# debug scripts use the same database unless .env is absent.
load_dotenv(override=True)

def _get_config_value(key: str) -> str | None:
    value = os.getenv(key)
    if value is not None and str(value).strip():
        return str(value).strip()

    try:
        secret_value = st.secrets.get(key)
    except Exception:
        secret_value = None

    if secret_value is None:
        return None

    secret_text = str(secret_value).strip()
    return secret_text if secret_text else None

def _require_env(key: str) -> str:
    value = _get_config_value(key)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {key}")
    return value.strip()

def _require_env_key(key: str) -> str:
    value = _get_config_value(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error
except ImportError:
    mysql_connector = None
    Error = Exception

def get_mysql_config() -> dict[str, object]:
    return {
        "host": _require_env("DB_HOST"),
        "port": int(_require_env("DB_PORT")),
        "user": _require_env("DB_USER"),
        "password": _require_env_key("DB_PASSWORD"),
        "database": _require_env("DB_NAME"),
    }

def connect_mysql():
    if mysql_connector is None:
        error_msg = "mysql-connector-python is not installed"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None

    try:
        connection = mysql_connector.connect(**get_mysql_config())
        st.session_state.pop("mysql_error", None)
        return connection
    except ValueError as exc:
        error_msg = f"Configuration error: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None
    except Error as exc:
        error_msg = f"Database connection failed: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None

def test_mysql_connection() -> bool:
    connection = connect_mysql()
    if connection is None:
        return False

    try:
        return connection.is_connected()
    finally:
        if connection is not None:
            connection.close()

def get_or_create_user_id(session_id: str, connection=None) -> int | None:
    """Get or create a user ID for the given session.
    
    Args:
        session_id: The session identifier
        connection: Optional existing database connection. If not provided, a new one will be created.
        
    Returns:
        User ID if successful, None if there was an error
    """
    close_connection = False
    if connection is None:
        connection = connect_mysql()
        close_connection = True
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
            st.session_state.pop("mysql_error", None)
            return row[0]

        # Make user creation idempotent so Streamlit reruns cannot trip the
        # unique constraint on users.session_id between SELECT and INSERT.
        cursor.execute(
            """
            INSERT INTO users (session_id)
            VALUES (%s)
            ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id)
            """,
            (session_id,),
        )
        connection.commit()
        user_id = cursor.lastrowid
        st.session_state.pop("mysql_error", None)
        print(f"[DEBUG] Created new user with ID {user_id} for session {session_id}")
        return user_id
    except Error as exc:
        error_msg = f"Failed to get or create user: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if close_connection and connection is not None:
            connection.close()

def save_resume_upload(session_id: str, file_name: str) -> int | None:
    connection = connect_mysql()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        # Pass the existing connection to avoid creating duplicate connections
        user_id = get_or_create_user_id(session_id, connection=connection)
        if user_id is None:
            error_msg = "Could not create or load user row. Check database connection and users table."
            st.session_state["mysql_error"] = error_msg
            print(f"[ERROR] {error_msg}")
            return None

        cursor.execute(
            """
            INSERT INTO resumes (user_id, file_name, chunk_count)
            VALUES (%s, %s, %s)
            """,
            (user_id, file_name, 1),
        )
        connection.commit()
        resume_id = cursor.lastrowid
        print(f"[DEBUG] Saved resume upload: user_id={user_id}, file={file_name}, resume_id={resume_id}")
        return resume_id
    except Error as exc:
        error_msg = f"Failed to save resume upload: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def save_resume_text(resume_id: int, resume_text: str) -> None:
    connection = connect_mysql()
    if connection is None:
        return

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO resume_chunks (resume_id, chunk_index, chunk_text)
            VALUES (%s, %s, %s)
            """,
            (resume_id, 0, resume_text),
        )
        connection.commit()
        print(f"[DEBUG] Saved resume text for resume_id={resume_id}")
    except Error as exc:
        error_msg = f"Failed to save resume text: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def load_resume_text(session_id: str, file_name: str | None = None) -> str:
    connection = connect_mysql()
    if connection is None:
        return ""

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
        return "\n\n".join(row[0] for row in rows if row[0])
    except Error as exc:
        error_msg = f"Failed to load resume text: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return ""
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def load_latest_resume_reference(session_id: str) -> tuple[int, str] | None:
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
            JOIN users u ON r.user_id = u.id
            WHERE u.session_id = %s
            ORDER BY r.uploaded_at DESC, r.id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row[0], row[1]
    except Error as exc:
        error_msg = f"Failed to load latest resume reference: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def load_resume_text_for_latest_upload(session_id: str) -> str:
    latest_ref = load_latest_resume_reference(session_id)
    if latest_ref is None:
        return ""

    latest_resume_id, _ = latest_ref
    connection = connect_mysql()
    if connection is None:
        return ""

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
        return "\n\n".join(row[0] for row in cursor.fetchall() if row[0])
    except Error as exc:
        error_msg = f"Failed to load resume text for latest upload: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return ""
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def save_chat_message(session_id: str, role: str, content: str) -> None:
    connection = connect_mysql()
    if connection is None:
        return

    cursor = None
    try:
        cursor = connection.cursor()
        # Pass the existing connection to avoid creating duplicate connections
        user_id = get_or_create_user_id(session_id, connection=connection)
        if user_id is None:
            error_msg = "Could not create or load user row. Check database connection and users table."
            st.session_state["mysql_error"] = error_msg
            print(f"[ERROR] {error_msg}")
            return

        cursor.execute(
            """
            SELECT id
            FROM chat_sessions
            WHERE user_id = %s AND session_name = %s
            LIMIT 1
            """,
            (user_id, session_id),
        )
        existing_session = cursor.fetchone()
        if existing_session is None:
            cursor.execute(
                """
                INSERT INTO chat_sessions (user_id, session_name)
                VALUES (%s, %s)
                """,
                (user_id, session_id),
            )

        cursor.execute(
            """
            INSERT INTO chat_messages (session_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (session_id, role, content),
        )
        connection.commit()
        print(f"[DEBUG] Saved chat message: role={role}, user_id={user_id}")
    except Error as exc:
        error_msg = f"Failed to save chat message: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def load_chat_messages(session_id: str, limit: int = 50) -> list[dict[str, str]]:
    connection = connect_mysql()
    if connection is None:
        return []

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]
    except Error as exc:
        error_msg = f"Failed to load chat messages: {str(exc)}"
        st.session_state["mysql_error"] = error_msg
        print(f"[ERROR] {error_msg}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def process_document(uploaded_file: UploadedFile) -> str:
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

    full_text = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
    if not full_text:
        raise ValueError("No readable text was found in this PDF.")

    return full_text
