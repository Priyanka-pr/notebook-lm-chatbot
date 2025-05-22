

import os
from markitdown import MarkItDown


def extract_text_from_file(file_path) : 
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def convert_to_text(file_path):
    file_type = {'.pdf': 'pdf', '.docx': 'docx', '.xlsx': 'xlsx', '.csv': 'csv', '.txt': 'txt'}.get(os.path.splitext(file_path)[1].lower())
    # Step 1: Extract text from the file based on the file type
    if file_type in ['pdf', 'docx', 'xlsx', 'csv'] : 
        text = extract_text_from_file(file_path)
    elif file_type == 'txt':
        text = extract_text_from_txt(file_path)
    else:
        print("Unsupported file type.")
        return

    if not text:
        print(f"No text extracted from {file_path}.")
        return

    return text