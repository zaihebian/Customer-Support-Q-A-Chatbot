# utils.py
SYSTEM_PROMPT = """You are an assistant for Holland & Barrett style FAQs.
Use ONLY the provided context. If unsure, say you don't know.
Answer in simple, short sentences. Include source file names when helpful.
"""

def make_prompt(context, question):
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""