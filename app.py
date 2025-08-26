# app.py
import os, pathlib, warnings
import streamlit as st

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter # break docs into chunks with structure
from langchain.schema import Document  # standard wrapper for text + metadata
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings # using BAAI/bge-small for strong,free embeddings.

from transformers import pipeline, AutoTokenizer # standardized interface to generation models like Flan-T5
from sentence_transformers import CrossEncoder # reranker model fine-tuned for relevance scoring.

# these are practical, free,open-source defaults.

# Optional BM25 fallback
try:
    from langchain_community.retrievers import BM25Retriever
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------- UI ----------
st.set_page_config(page_title="Customer Support Q&A Chatbot", page_icon="\U0001F4AC")
st.title("Customer Support Q&A Chatbot")
#st.caption("RAG · bge-small (embeddings) · CrossEncoder rerank · FLAN-T5-base (generator)")
st.caption("Developed by Parker Bai")

# ---------- Paths & constants ----------
INDEX_DIR = "index/faiss"
DATA_DIR = pathlib.Path("data_clean")

TOPK_CANDIDATES = 10      # initial dense retrieval pool
CONF_THRESHOLD = 0.35     # reranker confidence for "I don't know."
DELTA_TOP1_TOP2 = 0.10    # if top1 - top2 > this, use only top1
TOKEN_BUDGET = 480        # keep prompt < 512 for flan-t5-base
SNIPPET_CHAR_CAP = 700    # pre-trim each block before token budgeting

# ---------- Index build ----------
def build_index():
    os.makedirs("index", exist_ok=True)
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    docs = []
    for p in DATA_DIR.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for d in md_splitter.split_text(text):
            section = " / ".join(
                filter(None, [d.metadata.get("h1",""), d.metadata.get("h2",""), d.metadata.get("h3","")])
            )
            docs.append(
                Document(
                    page_content=f"{section}\n\n{d.page_content}".strip(),
                    metadata={"source": p.name, "section": section},  # which document, and which part of the document
                )
            )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(INDEX_DIR)

# ---------- Cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    if not os.path.exists(INDEX_DIR):
        if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.md")):
            st.error("No Markdown files found in data_clean/. Add .md files and redeploy.")
            st.stop()
        build_index()
    emb = get_embeddings()
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner=False)
def get_generator_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer=tok,
        max_new_tokens=160,
        min_new_tokens=8,        # avoid empty outputs
        num_beams=4,             # stable decoding
        do_sample=False,
        no_repeat_ngram_size=3,  # reduce repetition
        truncation=True
    )
    return gen, tok


@st.cache_resource(show_spinner=False)
def get_bm25_retriever():
    if not HAVE_BM25:
        return None
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    corpus = []
    for p in DATA_DIR.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for d in md_splitter.split_text(text):
            section = " / ".join(filter(None, [d.metadata.get("h1",""), d.metadata.get("h2",""), d.metadata.get("h3","")]))
            corpus.append(
                Document(
                    page_content=(section + "\n\n" + d.page_content).strip(),
                    metadata={"source": p.name, "section": section},
                )
            )
    return BM25Retriever.from_documents(corpus)

# ---------- Helpers ----------
def build_prompt_with_budget(q, docs, tokenizer, token_budget=TOKEN_BUDGET):
    instr = (
        "Answer ONLY from the snippets below. Prefer [1]. "
        "If the answer is not present, reply exactly: I don't know. "
        "Write 1–2 short sentences.\n\n"
    )
    head = f"Question: {q}\n"
    header_tokens = len(tokenizer.encode(instr + head, add_special_tokens=False))

    context = ""
    used = header_tokens
    for i, d in enumerate(docs, 1):
        block_text = d.page_content[:SNIPPET_CHAR_CAP]
        block = f"[{i}] Source: {d.metadata.get('source')} — {d.metadata.get('section')}\n{block_text}"
        btoks = len(tokenizer.encode("\n\n---\n\n" + block, add_special_tokens=False))
        if used + btoks > token_budget:
            break
        context += "\n\n---\n\n" + block
        used += btoks

    return instr + head + "Context:\n" + context + "\n\nAnswer:"

# ---------- Load resources ----------
vs = get_vectorstore()
reranker = get_reranker()
generator, tokenizer = get_generator_and_tokenizer()
bm25 = get_bm25_retriever()

# ---------- App ----------
st.subheader('Please ask a question')
q = st.text_input(" ")

if q:
    with st.spinner("Retrieving…"):
        candidates = vs.similarity_search(q, k=TOPK_CANDIDATES)

    if not candidates:
        st.subheader("Answer")
        st.write("I don't know.")
        st.stop()

    # Rerank by cross-encoder relevance
    pairs = [(q, d.page_content[:1000]) for d in candidates]
    scores = reranker.predict(pairs)  # higher = better
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    top1_score, top1_doc = ranked[0]
    top2_score = ranked[1][0] if len(ranked) > 1 else -1.0

    # Confidence gate
    use_docs = None
    if top1_score < CONF_THRESHOLD and bm25 is not None:
        # BM25 keyword fallback
        # bm_hits = bm25.get_relevant_documents(q)[:2]
        # if bm_hits:
        #     use_docs = bm_hits
        #     top1_score = float("nan")  # indicate fallback
        st.subheader("Answer")
        st.write("I don't know.")
        st.stop()
    elif top1_score < CONF_THRESHOLD:
        st.subheader("Answer")
        st.write("I don't know.")
        # with st.expander("Why?"):
        #     st.write(f"Top reranker score too low ({top1_score:.3f} < {CONF_THRESHOLD}).")
        st.stop()

    # If confident: choose top-1 or top-2 based on margin
    if use_docs is None:
        if (top1_score - top2_score) > DELTA_TOP1_TOP2:
            use_docs = [top1_doc]
        else:
            use_docs = [d for _, d in ranked[:2]]

    # Build prompt within token budget
    prompt = build_prompt_with_budget(q, use_docs, tokenizer)

    # with st.expander("Debug (scores & prompt)"):
    #     st.write(f"Top reranker score: {top1_score if top1_score==top1_score else 'BM25 fallback'}")
    #     st.text_area("Prompt (truncated to budget)", prompt, height=200)

    # Generate
    try:
        out = generator(prompt)[0]["generated_text"].strip()
    except Exception as e:
        st.error("Model generation failed.")
        st.exception(e)
        st.stop()

    if not out:
        out = "I don't know."

    st.subheader("Answer")
    st.write(out)

    # with st.expander("Sources"):
    #     for i, d in enumerate(use_docs, 1):
    #         st.write(f"- [{i}] {d.metadata.get('source')} — {d.metadata.get('section')}")
