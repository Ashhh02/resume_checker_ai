# Resume Analysis and Career Recommendation System

Streamlit app for resume analysis, job-fit checking, and career guidance using a local LLM (Ollama), LangChain chunking, and Cross-Encoder reranking.  
Data persistence is MySQL-backed for users, uploaded resumes, resume chunks, and chat history.

## Core Functions

- Upload and parse PDF resumes
- Split resume content into chunks
- Store resume metadata/chunks in MySQL
- Ask resume-aware questions in chat
- Save chat messages to MySQL
- Compare resume context against a pasted/fetched job description

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- AI/LLM: Ollama (`llama3.2:3b`)
- NLP Pipeline: LangChain + PyMuPDF
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Database: MySQL

## Prerequisites

- Python 3.10+
- MySQL server running
- Ollama installed and running locally

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pull Ollama model:
```bash
ollama pull llama3.2:3b
```

3. Create `.env` in the project root:
```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=resume_ai
```

4. Ensure your MySQL schema/tables already exist in `resume_ai`:
- `users`
- `resumes`
- `resume_chunks`
- `chat_sessions`
- `chat_messages`

## Run

```bash
streamlit run resume.py
```

## How to Use

1. Upload a resume PDF in the sidebar.
2. Click `Process Resume`.
3. Optionally paste a job URL and click `Fetch Job Description`, or paste job text directly.
4. Ask questions in the chat (score, missing skills, rewrite suggestions, role fit, next actions).

## Notes

- This project uses `.env` values for MySQL connection.
- `.env` is ignored by git (`.gitignore`) to protect credentials.
- If you see `1045 Access denied`, verify `.env` username/password and restart Streamlit.
