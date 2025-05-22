import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from typing import Dict, List, Tuple, Union
import numpy as np

class PageContentExtractor:
    def __init__(self, languages=['en']):
        """
        Initialize the page content extractor with OCR reader
        
        Args:
            languages (list): List of languages for OCR recognition
        """
        self.ocr_reader = easyocr.Reader(languages)
    
    def extract_page_content(self, page, page_number: int = None) -> Dict[str, Union[str, List[str], bool]]:
        """
        Extract content from a page - handles text, images, or both
        
        Args:
            page: PDF page object (PyMuPDF page object)
            page_number (int): Page number for reference
            
        Returns:
            dict: Contains extracted text, OCR text, and metadata
        """
        result = {
            'page_number': page_number,
            'has_text': False,
            'has_images': False,
            'extracted_text': '',
            'ocr_text': [],
            'total_content': ''
        }
        
        # Extract regular text from page
        page_text = page.get_text().strip()
        
        # Check if page has readable text
        if page_text and len(page_text) > 10:  # Minimum threshold for meaningful text
            result['has_text'] = True
            result['extracted_text'] = page_text
        
        # Extract images from page
        image_list = page.get_images()
        
        if image_list:
            result['has_images'] = True
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("ppm")
                        image = Image.open(io.BytesIO(img_data))
                    else:  # CMYK: convert to RGB
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("ppm")
                        image = Image.open(io.BytesIO(img_data))
                        pix1 = None
                    
                    pix = None
                    
                    # Convert PIL Image to numpy array for OCR
                    img_array = np.array(image)
                    
                    # Perform OCR on the image
                    ocr_results = self.ocr_reader.readtext(img_array)
                    
                    # Extract text from OCR results
                    image_text = ' '.join([result[1] for result in ocr_results if result[2] > 0.5])  # Confidence > 0.5
                    
                    if image_text.strip():
                        result['ocr_text'].append({
                            'image_index': img_index,
                            'text': image_text.strip()
                        })
                
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_number}: {str(e)}")
                    continue
        
        # Combine all extracted content
        all_content = []
        
        if result['has_text']:
            all_content.append(result['extracted_text'])
        
        if result['ocr_text']:
            for ocr_item in result['ocr_text']:
                all_content.append(ocr_item['text'])
        
        result['total_content'] = '\n\n'.join(all_content)
        
        return result
    
    def process_pdf_file(self, pdf_path: str) -> List[Dict]:
        """
        Process entire PDF file page by page
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries containing extracted content for each page
        """
        results = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = self.extract_page_content(page, page_num + 1)
                results.append(page_content)
                
                # Print progress
                print(f"Processed page {page_num + 1}/{len(doc)}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            
        return results
    
    def get_page_summary(self, page_result: Dict) -> str:
        """
        Get a summary of what was found on the page
        
        Args:
            page_result (dict): Result from extract_page_content
            
        Returns:
            str: Summary string
        """
        page_num = page_result.get('page_number', 'Unknown')
        
        if page_result['has_text'] and page_result['has_images']:
            return f"Page {page_num}: Contains both text and {len(page_result['ocr_text'])} image(s) with text"
        elif page_result['has_text']:
            return f"Page {page_num}: Contains only readable text"
        elif page_result['has_images']:
            return f"Page {page_num}: Contains only {len(page_result['ocr_text'])} image(s) with text"
        else:
            return f"Page {page_num}: No readable content found"

# # Example usage
# def main():
#     # Initialize the extractor
#     extractor = PageContentExtractor(languages=['en'])  # Add more languages as needed
    
#     # Process a single page (if you have a page object)
#     # page_result = extractor.extract_page_content(page, page_number=1)
    
#     # Or process an entire PDF file
#     pdf_path = "your_document.pdf"  # Replace with your PDF path
#     all_results = extractor.process_pdf_file(pdf_path)
    
#     # Display results
#     for result in all_results:
#         print("=" * 50)
#         print(extractor.get_page_summary(result))
#         print("-" * 30)
        
#         if result['total_content']:
#             print("Extracted Content:")
#             print(result['total_content'][:500] + "..." if len(result['total_content']) > 500 else result['total_content'])
#         else:
#             print("No content extracted from this page")
        
#         print()

# if __name__ == "__main__":
#     main()