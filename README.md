# Resume & Career Advisor Chatbot

A Streamlit-based AI chatbot that analyzes your resume and provides personalized career advice using local LLMs and RAG (Retrieval-Augmented Generation).

## Prerequisites

- Python 3.10+
- Ollama installed and running locally

## Setup

1. **Install Ollama** (if not already installed):
   - Download from [ollama.ai](https://ollama.ai)
   - Start the Ollama service

2. **Pull required models**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the app**:
   ```bash
   streamlit run resume.py
   ```

2. **How to use**:
   - Upload your resume PDF in the sidebar
   - Click "Process Resume" to load it
   - Ask questions about your resume or career in the chat interface
   - Use quick question buttons for common queries

## Features

- **Resume Analysis**: Get detailed feedback on strengths, weaknesses, and improvements
- **Career Advice**: Receive tailored recommendations based on your experience
- **Chat Interface**: Conversational AI with chat history
- **Local Processing**: Everything runs locally - no data sent to the cloud
- **RAG Technology**: Uses vector search and re-ranking for accurate responses

## Tech Stack

- **Frontend**: Streamlit
- **Vector DB**: ChromaDB with Ollama embeddings
- **LLM**: Llama 3.2 3B via Ollama
- **PDF Processing**: PyMuPDF + LangChain
- **Re-ranking**: Sentence Transformers CrossEncoder

## Note

This app is fully local and privacy-focused. Your resume data never leaves your machine, and no API keys are required.