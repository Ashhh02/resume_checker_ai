import ollama
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import CrossEncoder
from resume_db import load_resume_chunks, load_resume_chunks_for_latest_upload

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
        docs = load_resume_chunks_for_latest_upload(session_id)
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
