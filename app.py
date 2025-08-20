import os, pathlib
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "index/faiss"
DATA_DIR = pathlib.Path("data_clean")

def ensure_index():
    if os.path.exists(INDEX_DIR):
        return
    os.makedirs("index", exist_ok=True)
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    docs_by_section = []
    for p in DATA_DIR.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for d in md_splitter.split_text(text):
            section = " / ".join(filter(None, [d.metadata.get("h1",""), d.metadata.get("h2",""), d.metadata.get("h3","")]))
            docs_by_section.append(
                Document(
                    page_content=f"{section}\n\n{d.page_content}".strip(),
                    metadata={"source": p.name, "section": section},
                )
            )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    splits = splitter.split_documents(docs_by_section)
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(INDEX_DIR)

ensure_index()



# app.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="H&B FAQ Chatbot", page_icon="💬")

# --- load vector index + embeddings (same as you used to build) ---
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
vs = FAISS.load_local("index/faiss", emb, allow_dangerous_deserialization=True)

# --- tiny local LLM (free) ---
# flan-t5-small is light; good enough for short, grounded answers
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=128
)

st.title("H&B FAQ Chatbot")
st.caption("Free + local: bge-small embeddings + FLAN-T5-small")

q = st.text_input("Ask a question about delivery, returns, marketplace, etc.")

if q:
    # retrieve diverse, relevant chunks
    docs = vs.max_marginal_relevance_search(q, k=5, fetch_k=25)

    # build context (keep it short)
    context = "\n\n---\n\n".join(d.page_content[:800] for d in docs)

    # prompt the small model to answer ONLY from context
    prompt = (
        "Answer the question using ONLY the context. If not in context, say \"I don't know.\" "
        "Be concise.\n\n"
        f"Question: {q}\n\n"
        f"Context:\n{context}\n\nAnswer:"
    )
    out = generator(prompt)[0]["generated_text"].strip()

    st.subheader("Answer")
    st.write(out)

    with st.expander("Sources"):
        for d in docs:
            st.write(f"- {d.metadata.get('source')}")


## streamlit run app.py --server.port 8501 --server.address 0.0.0.0