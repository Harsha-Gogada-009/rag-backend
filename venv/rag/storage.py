import json
import os
from datetime import datetime

CLEANED_TEXT_FILE = "cleaned_text.json"

def save_cleaned_file(
    file_name: str,
    file_type: str,
    cleaned_text: str
):
    entry = {
        "file_name": file_name,
        "file_type": file_type,
        "timestamp": datetime.now().isoformat(),
        "cleaned_text": cleaned_text
    }

    if os.path.exists(CLEANED_TEXT_FILE):
        with open(CLEANED_TEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(CLEANED_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
