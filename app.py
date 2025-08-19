# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import make_prompt

load_dotenv()
st.set_page_config(page_title="H&B FAQ Chatbot", page_icon="💬")

# Load index and models
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.load_local("index/faiss", emb, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

st.title("H&B Business Chatbot")
st.caption("Ask about returns, delivery, product info, and policies.")

if "history" not in st.session_state: st.session_state.history = []

q = st.text_input("Your question")
if st.button("Ask") or q:
    if not q: st.stop()
    docs = vs.similarity_search(q, k=4)
    context = "\n\n---\n\n".join(d.page_content[:1200] for d in docs)
    prompt = make_prompt(context, q)
    with st.spinner("Thinking..."):
        resp = llm.invoke(prompt).content

    st.session_state.history.append(("You", q))
    st.session_state.history.append(("Bot", resp))

# Show chat
for role, msg in st.session_state.history:
    st.markdown(f"**{role}:** {msg}")

# Feedback
st.subheader("Feedback")
col1, col2 = st.columns(2)
with col1:
    good = st.button("👍 Helpful")
with col2:
    bad = st.button("👎 Not helpful")

if good or bad:
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/log.csv","a",encoding="utf-8") as f:
        f.write(f"{q}\t{'good' if good else 'bad'}\n")
    st.success("Thanks!")