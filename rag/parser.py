import fitz  # PyMuPDF

def parse_pdf(path: str) -> str:
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text


def parse_python(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
