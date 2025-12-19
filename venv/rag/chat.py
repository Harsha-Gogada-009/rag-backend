from openai import OpenAI
from dotenv import load_dotenv

from rag.retriever import Retriever

load_dotenv()

client = OpenAI()
retriever = Retriever()


def rag_chat(query: str) -> str:
    retrieved = retriever.retrieve(query)

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
