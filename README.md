**A production-ready Retrieval-Augmented Generation (RAG) chatbot that turns Markdown documentation into a conversational assistant.
Built to demonstrate how LLMs can deliver trustworthy, real-time answers from structured company knowledge.**

## 🚀 Features

- End-to-End RAG Pipeline
FAISS dense retrieval → CrossEncoder reranking → FLAN-T5 generation

- Trustworthy Answers
Confidence thresholds + fallback search + “I don’t know” guardrails

- Efficient Design
Token budgeting, caching, and modular components for easy model swaps

- Interactive UI
Streamlit app with real-time Q&A interface

- Extensible
Swap embeddings, vector DB, or generator models with minimal changes

## 🛠️ Tech Stack

- Python
- Streamlit (UI)
- FAISS (vector store)
- Hugging Face Transformers (embeddings + generation)
- CrossEncoder (reranking)
- BM25 (keyword fallback)

## 📂 Project Workflow

- Data Preparation
- Parse Markdown files
- Chunk text (~800 chars, 150 overlap)
- Embed with bge-small-en v1.5 → store in FAISS
- Retrieval
- FAISS similarity search for top candidates
- CrossEncoder reranks for best relevance
- Answer Generation
- Build context-aware prompt (with token budget)
- Generate answer with google/flan-t5-base
- Fall back to BM25 or return “I don’t know” if low confidence
- User Interface
- Streamlit app shows answer + sources
- Debug panel with scores and prompt trace

## ⚡ Setup & Run

- Clone the repo:
``` git clone https://github.com/your-username/your-repo.git ```
``` cd your-repo ```
- Create environment & install dependencies:
``` pip install -r requirements.txt ```
- Launch the app:
``` streamlit run app.py ```

## 🔧 Configuration
- Indexing:
Place .md files in data_clean/ and rebuild the FAISS index.
- Thresholds:
Adjust confidence (CONF_THRESHOLD) and snippet limits in constants.py.
- Secrets/API Keys:
Add in .streamlit/secrets.toml if needed for external models.

## 🌍 Future Improvements

- Deploy backend via FastAPI + managed vector DB (e.g., Pinecone)
- GPU-optimized inference with vLLM or SageMaker
- Containerization & autoscaling with Docker + Kubernetes
- Add monitoring and feedback loop for continuous improvement
