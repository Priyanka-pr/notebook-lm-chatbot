

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
    # print("hello")
    md = MarkItDown()
    result = md.convert(file_path)
    print("hello")
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
    # print(f"text chunks ::: {text}")

    return text

def convert_to_text_1(file_path):
        # Initialize the extractor with ChromaDB
    extractor = PageContentExtractor(
        languages=['en'],
        chroma_db_path="./document_chroma_db",
        collection_name="pdf_content"
    )
    
    # Process a PDF file
    if os.path.exists(file_path):
        results = extractor.process_pdf_file(file_path, save_to_db=True)
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        total_saved = sum(len(result['saved_items']) for result in results)
        print(f"Total items saved to ChromaDB: {total_saved}")
        
        # Show collection statistics
        stats = extractor.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Example search
        print("\n" + "="*60)
        print("SEARCH EXAMPLE")
        print("="*60)
        
        search_results = extractor.search_similar_content("your search query here", n_results=3)
        if 'error' not in search_results:
            print(f"Found {search_results['count']} similar items")
            for i, doc in enumerate(search_results['results']['documents'][0]):
                metadata = search_results['results']['metadatas'][0][i]
                print(f"{i+1}. {metadata['content_type']} from page {metadata['page_number']}")
                print(f"   Preview: {doc[:100]}...")
                print()