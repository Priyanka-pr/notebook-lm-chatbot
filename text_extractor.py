

import os
from markitdown import MarkItDown
from PageContentExtractor import PageContentExtractor


def extract_text_from_file_2(file_path) :
    # Initialize the extractor
    extractor = PageContentExtractor(languages=['en'])  # Add more languages as needed
    
    # Process a single page (if you have a page object)
    # page_result = extractor.extract_page_content(page, page_number=1)
    
    # Or process an entire PDF file
    all_results = extractor.process_pdf_file(file_path)
    
    # Display results
    for result in all_results:
        print("=" * 50)
        print(extractor.get_page_summary(result))
        print("-" * 30)
        
        if result['total_content']:
            print("Extracted Content:")
            print(result['total_content'][:500] + "..." if len(result['total_content']) > 500 else result['total_content'])
        else:
            print("No content extracted from this page")
            
    return all_results

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