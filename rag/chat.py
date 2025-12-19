from openai import OpenAI
from dotenv import load_dotenv

from rag.retriever import Retriever

load_dotenv()

client = OpenAI()
_retriever = None


def _get_retriever():
    """Lazy initialization of retriever to avoid loading index at module import time."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def rag_chat(query: str) -> str:
    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    
    try:
        retrieved = retriever.retrieve(query)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

    context_blocks = []
    for r in retrieved:
        meta = r["meta"]
        block = f"""
File: {meta['file_name']} ({meta['file_type']})
Time: {meta['timestamp']}
Content:
{r['chunk']}
"""
        context_blocks.append(block)

    context = "\n---\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant answering questions based on the following documents.

Document context:
{context}

User question:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content
