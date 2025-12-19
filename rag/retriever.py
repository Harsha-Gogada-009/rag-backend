import json
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_FILE = "faiss_index.index"
CHUNKS_FILE = "chunks.json"


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX_FILE)

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = np.array(
            self.embeddings.embed_query(query)
        ).astype("float32")

        query_vec = np.expand_dims(query_vec, axis=0)

        _, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "chunk": self.chunks[idx],
                "meta": self.metadata[idx]
            })

        return results
