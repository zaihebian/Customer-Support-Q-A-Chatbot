# rag_test.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vs = FAISS.load_local("index/faiss", emb, allow_dangerous_deserialization=True)

while True:
    q = input("\nQuestion (or 'q' to quit): ").strip()
    if q.lower() == "q":
        break
    docs = vs.similarity_search(q, k=4)
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] Source: {d.metadata.get('source')}")
        print(d.page_content[:500], "...")