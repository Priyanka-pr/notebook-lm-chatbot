import os
import PyPDF2
from docx import Document
import pandas as pd

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX using python-docx
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from XLSX using pandas
def extract_text_from_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path)
    text = df.to_string(index=False)  # Converts the DataFrame to a text representation
    return text

# Function to extract text from CSV using pandas
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string(index=False)  # Converts the DataFrame to a text representation
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def convert_to_text(file_path):
    file_type = {'.pdf': 'pdf', '.docx': 'docx', '.xlsx': 'xlsx', '.csv': 'csv', '.txt': 'txt'}.get(os.path.splitext(file_path)[1].lower())
    # Step 1: Extract text from the file based on the file type
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_type == 'xlsx':
        text = extract_text_from_xlsx(file_path)
    elif file_type == 'csv':
        text = extract_text_from_csv(file_path)
    elif file_type == 'txt':
        text = extract_text_from_txt(file_path)
    else:
        print("Unsupported file type.")
        return

    if not text:
        print(f"No text extracted from {file_path}.")
        return

    return text