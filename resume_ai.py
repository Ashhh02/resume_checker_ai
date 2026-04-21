import os
import re
import json

import ollama
import requests
import streamlit as st
from bs4 import BeautifulSoup
from resume_db import load_resume_text, load_resume_text_for_latest_upload

# Use remote Ollama via Together.ai
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

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
- Prioritize the most relevant resume snippets for the user's question and the optional job description
- If a detail is not present in the resume context, say it is not shown instead of guessing
"""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "i", "if", "in", "into", "is", "it", "me", "my", "of", "on", "or",
    "our", "the", "to", "we", "what", "when", "where", "which", "with", "you",
    "your",
}

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

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9\+#\./-]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]

def _split_resume_sections(resume_text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", resume_text) if block.strip()]
    if len(blocks) > 1:
        return blocks

    sentence_blocks = re.split(r"(?<=[.!?])\s+(?=[A-Z])", resume_text)
    merged_sections: list[str] = []
    current = ""
    for block in sentence_blocks:
        cleaned = _normalize_whitespace(block)
        if not cleaned:
            continue
        candidate = f"{current} {cleaned}".strip() if current else cleaned
        if len(candidate) <= 500:
            current = candidate
        else:
            if current:
                merged_sections.append(current)
            current = cleaned

    if current:
        merged_sections.append(current)

    return merged_sections or [_normalize_whitespace(resume_text)]

def _score_section(section: str, query_tokens: list[str], preferred_tokens: set[str]) -> int:
    section_tokens = _tokenize(section)
    if not section_tokens:
        return 0

    token_set = set(section_tokens)
    score = 0
    for token in query_tokens:
        if token in token_set:
            score += 3
    for token in preferred_tokens:
        if token in token_set:
            score += 2

    score += min(len(section_tokens), 120) // 20
    return score

def build_resume_context(
    prompt: str,
    resume_text: str,
    job_description: str | None = None,
    max_sections: int = 4,
) -> tuple[str, list[str]]:
    cleaned_resume = resume_text.strip()
    if not cleaned_resume:
        return "", []

    sections = _split_resume_sections(cleaned_resume)
    query_text = prompt.strip()
    if job_description and job_description.strip():
        query_text = f"{query_text}\n{job_description.strip()[:1500]}"

    query_tokens = _tokenize(query_text)
    preferred_tokens = set(query_tokens[:16])

    ranked_sections = sorted(
        sections,
        key=lambda section: _score_section(section, query_tokens, preferred_tokens),
        reverse=True,
    )

    selected_sections: list[str] = []
    seen_normalized: set[str] = set()
    for section in ranked_sections:
        normalized = _normalize_whitespace(section).lower()
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        selected_sections.append(_normalize_whitespace(section))
        if len(selected_sections) >= max_sections:
            break

    if not selected_sections:
        selected_sections = [_normalize_whitespace(cleaned_resume[:2000])]

    labeled_sections = [
        f"[Resume snippet {index + 1}]\n{section}"
        for index, section in enumerate(selected_sections)
    ]
    return "\n\n".join(labeled_sections), selected_sections

def search_resume_context(
    prompt: str,
    session_id: str,
    file_name: str | None = None,
    job_description: str | None = None,
) -> tuple[str, list[str]]:
    resume_text = load_resume_text(session_id, file_name)
    if not resume_text.strip():
        resume_text = load_resume_text_for_latest_upload(session_id)

    return build_resume_context(prompt, resume_text, job_description)

def extract_score(text: str) -> int | None:
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

def call_llm(
    context: str,
    prompt: str,
    job_description: str | None = None,
    model: str = DEFAULT_MODEL,
):
    user_content = (
        "Use the resume context below to answer the request.\n\n"
        f"Resume Context:\n{context}\n\n"
    )
    if job_description and job_description.strip():
        user_content += f"Job Description:\n{job_description.strip()}\n\n"
    user_content += f"User Request:\n{prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Models like qwen2.5:7b-instruct and llama3.2:3b are local Ollama models.
    # Only send provider-qualified models such as "meta-llama/..." to Together.
    if "/" not in model:
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.2,
                },
            )
            for chunk in response:
                message = chunk.get("message", {})
                content = message.get("content", "")
                if content:
                    yield content
            return
        except Exception as exc:
            raise RuntimeError(f"Local Ollama error for model '{model}': {exc}") from exc

    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        try:
            together_api_key = str(st.secrets.get("TOGETHER_API_KEY", "")).strip()
        except Exception:
            together_api_key = ""
    if not together_api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")

    try:
        with requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {together_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Together.ai request failed with status {response.status_code}: "
                    f"{response.text[:500]}"
                )

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content", "")
                if content:
                    yield content
    except Exception as exc:
        raise RuntimeError(f"Together.ai API error: {exc}") from exc
