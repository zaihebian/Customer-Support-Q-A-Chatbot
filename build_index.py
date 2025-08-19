# build_index.py
import pathlib, os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # <- new home

DATA = pathlib.Path("data_clean")
INDEX_DIR = "index/faiss"
os.makedirs("index", exist_ok=True)

# load docs
docs = []
for p in DATA.glob("*.md"):
    text = p.read_text(encoding="utf-8")
    docs.append(Document(page_content=text, metadata={"source": p.name}))

# chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = splitter.split_documents(docs)

# HF embeddings (free, local)
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}  # good for cosine search
)

# build index
vs = FAISS.from_documents(splits, emb)
vs.save_local(INDEX_DIR)
print("✅ Index saved to", INDEX_DIR, "with", len(splits), "chunks")
