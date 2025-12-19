import json
import numpy as np
import faiss
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from rag.chunker import chunk_text

load_dotenv()

CLEANED_TEXT_FILE = "cleaned_text.json"
CHUNKS_FILE = "chunks.json"
FAISS_INDEX_FILE = "faiss_index.index"


def build_index():
    if not os.path.exists(CLEANED_TEXT_FILE):
        raise FileNotFoundError("cleaned_text.json not found")

    with open(CLEANED_TEXT_FILE, "r", encoding="utf-8") as f:
        all_files = json.load(f)

    all_chunks = []
    chunk_metadata = []

    for file in all_files:
        chunks = chunk_text(file["cleaned_text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                "file_name": file["file_name"],
                "file_type": file["file_type"],
                "timestamp": file["timestamp"]
            })

    # Generate embeddings
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    embeddings = embeddings_model.embed_documents(all_chunks)
    embedding_array = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)

    # Save index
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save chunks + metadata
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": all_chunks,
                "metadata": chunk_metadata
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    return {
        "chunks": len(all_chunks),
        "vectors": index.ntotal
    }
