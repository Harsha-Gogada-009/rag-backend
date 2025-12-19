from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag.indexer import build_index
from fastapi import UploadFile, File
import os
import shutil
from pydantic import BaseModel
from rag.parser import parse_pdf, parse_python
from rag.cleaner import clean_text
from rag.storage import save_cleaned_file


class ChatRequest(BaseModel):
    query: str

app = FastAPI(title="DocuMind RAG Backend")

# Allow frontend (we'll tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def health():
    return {"status": "Backend running"}




from rag.chat import rag_chat


@app.post("/chat")
def chat(request: ChatRequest):
    answer = rag_chat(request.query)
    return {"answer": answer}




@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    os.makedirs("uploads", exist_ok=True)

    for file in files:
        file_path = os.path.join("uploads", file.filename)

        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Parse
        if file.filename.lower().endswith(".pdf"):
            raw_text = parse_pdf(file_path)
            file_type = "PDF"
        elif file.filename.lower().endswith(".py"):
            raw_text = parse_python(file_path)
            file_type = "Python (.py)"
        else:
            continue

        # Clean
        cleaned = clean_text(raw_text)

        # Store
        save_cleaned_file(
            file_name=file.filename,
            file_type=file_type,
            cleaned_text=cleaned
        )

    return {"message": "Files processed successfully"}




@app.post("/build-index")
def build_faiss_index():
    result = build_index()
    return {
        "message": "FAISS index built successfully",
        "details": result
    }
