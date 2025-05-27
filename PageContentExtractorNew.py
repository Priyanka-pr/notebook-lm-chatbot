import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import re
import os
import base64
import hashlib
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

class PageContentExtractorNew:
    def __init__(self, 
                 languages=['en'], 
                 chroma_db_path="./chroma_db",
                 collection_name="document_content",
                 text_model_name="all-MiniLM-L6-v2",
                 image_model_name="clip-ViT-B-32"):
        """
        Initialize the page content extractor with OCR reader and ChromaDB
        
        Args:
            languages (list): List of languages for OCR recognition
            chroma_db_path (str): Path to ChromaDB storage
            collection_name (str): Name of the ChromaDB collection
            text_model_name (str): Sentence transformer model for text embeddings
            image_model_name (str): Model for image embeddings
        """
        self.ocr_reader = easyocr.Reader(languages)
        
        # Initialize embedding models
        self.text_model = SentenceTransformer(text_model_name)
        
        # For image embeddings, we'll use CLIP
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            print("Warning: transformers library not found. Image embeddings will use text model on OCR text.")
            self.image_model = None
            self.image_processor = None
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection_name = collection_name
        
        # Get embedding dimensions for consistency
        sample_text_embedding = self.text_model.encode("test", convert_to_tensor=False)
        self.text_embedding_dim = len(sample_text_embedding)
        
        if self.image_model is not None:
            # Get CLIP embedding dimension
            test_image = Image.new('RGB', (224, 224), color='white')
            test_embedding = self.generate_image_embedding(test_image)
            self.image_embedding_dim = len(test_embedding) if test_embedding else self.text_embedding_dim
        else:
            self.image_embedding_dim = self.text_embedding_dim
        
        print(f"Text embedding dimension: {self.text_embedding_dim}")
        print(f"Image embedding dimension: {self.image_embedding_dim}")
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Document content with text and image embeddings"}
            )
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text content"""
        if not text.strip():
            return []
        
        embedding = self.text_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate embedding for image content"""
        if self.image_model is None:
            # Fallback: use OCR text for embedding
            img_array = np.array(image)
            ocr_results = self.ocr_reader.readtext(img_array)
            ocr_text = ' '.join([result[1] for result in ocr_results if result[2] > 0.5])
            return self.generate_text_embedding(ocr_text) if ocr_text else []
        
        try:
            # Use CLIP model for image embeddings
            inputs = self.image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)
            
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.squeeze().tolist()
        
        except Exception as e:
            print(f"Error generating image embedding: {str(e)}")
            return []
    
    def pad_or_truncate_embedding(self, embedding: List[float], target_dim: int) -> List[float]:
        """Ensure embedding has the correct dimension"""
        if not embedding:
            return []
        
        if len(embedding) == target_dim:
            return embedding
        elif len(embedding) > target_dim:
            # Truncate
            return embedding[:target_dim]
        else:
            # Pad with zeros
            return embedding + [0.0] * (target_dim - len(embedding))
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        # Ensure image is in RGB mode to avoid alpha channel issues
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def generate_content_id(self, content: str, content_type: str, page_num: int, index: int = 0) -> str:
        """Generate unique ID for content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{content_type}_page{page_num}_{index}_{content_hash}"
    
    def save_to_chromadb(self, 
                        content: str, 
                        embedding: List[float], 
                        metadata: Dict, 
                        content_id: str,
                        content_type: str = 'text'):
        """Save content and embedding to ChromaDB"""
        if not embedding:
            print(f"Warning: Empty embedding for content ID {content_id}")
            return
        
        try:
            # Ensure embedding has correct dimension based on content type
            target_dim = self.text_embedding_dim if content_type == 'text' else self.text_embedding_dim
            embedding = self.pad_or_truncate_embedding(embedding, target_dim)
            
            if not embedding:
                print(f"Warning: Could not fix embedding dimension for content ID {content_id}")
                return
            
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[content_id]
            )
        except Exception as e:
            print(f"Error saving to ChromaDB: {str(e)}")
    
    def extract_page_content(self, 
                           page, 
                           page_number: int = None, 
                           document_name: str = "unknown_doc",
                           save_to_db: bool = True) -> Dict[str, Union[str, List[str], bool]]:
        """
        Extract content from a page and save to ChromaDB
        
        Args:
            page: PDF page object (PyMuPDF page object)
            page_number (int): Page number for reference
            document_name (str): Name of the document
            save_to_db (bool): Whether to save to ChromaDB
            
        Returns:
            dict: Contains extracted text, OCR text, and metadata
        """
        result = {
            'page_number': page_number,
            'document_name': document_name,
            'has_text': False,
            'has_images': False,
            'extracted_text': '',
            'ocr_text': [],
            'total_content': '',
            'saved_items': []  # Track what was saved to DB
        }
        
        timestamp = datetime.now().isoformat()
        
        # Extract regular text from page
        page_text = page.get_text().strip()
        
        # Check if page has readable text
        if page_text and len(page_text) > 10:  # Minimum threshold for meaningful text
            result['has_text'] = True
            result['extracted_text'] = page_text
            
            if save_to_db:
                # Generate text embedding
                text_embedding = self.generate_text_embedding(page_text)
                
                if text_embedding:
                    # Create metadata for text content
                    text_metadata = {
                        'content_type': 'text',
                        'document_name': document_name,
                        'page_number': page_number,
                        'timestamp': timestamp,
                        'word_count': len(page_text.split()),
                        'char_count': len(page_text)
                    }
                    
                    # Generate unique ID for text content
                    text_id = self.generate_content_id(page_text, 'text', page_number)
                    
                    # Save to ChromaDB
                    self.save_to_chromadb(page_text, text_embedding, text_metadata, text_id, 'text')
                    result['saved_items'].append({'type': 'text', 'id': text_id})
        
        # Extract images from page
        image_list = page.get_images()
        
        if image_list:
            result['has_images'] = True
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image with proper handling of different formats
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert pixmap to PIL Image
                        img_data = pix.tobytes("png")  # Use PNG format to handle alpha properly
                        image = Image.open(io.BytesIO(img_data))
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        pix1 = None
                    
                    # Convert to RGB if not already (removes alpha channel issues)
                    if image.mode != 'RGB':
                        # Handle different modes properly
                        if image.mode == 'RGBA':
                            # Create white background for transparent images
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                            image = background
                        else:
                            image = image.convert('RGB')
                    
                    pix = None
                    
                    # Convert PIL Image to numpy array for OCR
                    img_array = np.array(image)
                    
                    # Perform OCR on the image
                    ocr_results = self.ocr_reader.readtext(img_array)
                    
                    # Extract text from OCR results
                    image_text = ' '.join([result[1] for result in ocr_results if result[2] > 0.5])
                    
                    if image_text.strip():
                        result['ocr_text'].append({
                            'image_index': img_index,
                            'text': image_text.strip()
                        })
                    
                    if save_to_db:
                        # For images, use OCR text for embedding to maintain consistency
                        if image_text.strip():
                            image_embedding = self.generate_text_embedding(image_text.strip())
                        else:
                            # Use a default description if no OCR text
                            default_text = f"Image {img_index} from page {page_number}"
                            image_embedding = self.generate_text_embedding(default_text)
                        
                        if image_embedding:
                            # Convert image to base64 for storage
                            image_b64 = self.image_to_base64(image)
                            
                            # Create metadata for image content
                            image_metadata = {
                                'content_type': 'image',
                                'document_name': document_name,
                                'page_number': page_number,
                                'image_index': img_index,
                                'timestamp': timestamp,
                                'image_size': f"{image.width}x{image.height}",
                                'ocr_text': image_text.strip(),
                                'ocr_confidence': 'high' if any(r[2] > 0.8 for r in ocr_results) else 'medium',
                                'image_data': image_b64  # Store base64 encoded image
                            }
                            
                            # Use OCR text as document content, or image description
                            content_for_storage = image_text.strip() if image_text.strip() else f"Image {img_index} from page {page_number}"
                            
                            # Generate unique ID for image content
                            image_id = self.generate_content_id(content_for_storage, 'image', page_number, img_index)
                            
                            # Save to ChromaDB
                            self.save_to_chromadb(content_for_storage, image_embedding, image_metadata, image_id, 'image')
                            result['saved_items'].append({'type': 'image', 'id': image_id, 'ocr_text': image_text.strip()})
                
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
    
    def process_pdf_file(self, 
                        pdf_path: str, 
                        document_name: str = None,
                        save_to_db: bool = True) -> List[Dict]:
        """
        Process entire PDF file page by page and save to ChromaDB
        
        Args:
            pdf_path (str): Path to the PDF file
            document_name (str): Name for the document (defaults to filename)
            save_to_db (bool): Whether to save to ChromaDB
            
        Returns:
            list: List of dictionaries containing extracted content for each page
        """
        results = []
        
        if document_name is None:
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            doc = fitz.open(pdf_path)
            
            print(f"Processing PDF: {document_name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                if(page_num == 4):
                    page = doc[page_num]
                    page_content = self.extract_page_content(
                        page, 
                        page_num + 1, 
                        document_name,
                        save_to_db
                    )
                    results.append(page_content)
                    
                    # Print progress
                    saved_count = len(page_content['saved_items'])
                    print(f"Processed page {page_num + 1}/{len(doc)} - Saved {saved_count} items to DB")
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
        
        return results
    
    def search_similar_content(self, 
                             query: str, 
                             n_results: int = 5,
                             content_type: str = None) -> Dict:
        """
        Search for similar content in ChromaDB
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            content_type (str): Filter by content type ('text' or 'image')
            
        Returns:
            dict: Search results
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_text_embedding(query)
            
            if not query_embedding:
                return {'error': 'Could not generate query embedding'}
            
            # Ensure query embedding has correct dimension
            query_embedding = self.pad_or_truncate_embedding(query_embedding, self.text_embedding_dim)
            
            # Prepare where clause for filtering
            where_clause = {}
            if content_type:
                where_clause['content_type'] = content_type
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            return {
                'query': query,
                'results': results,
                'count': len(results['documents'][0]) if results['documents'] else 0
            }
            
        except Exception as e:
            return {'error': f'Search failed: {str(e)}'}
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            
            # Get some sample data to analyze
            sample_data = self.collection.get(limit=100)
            
            content_types = {}
            documents = {}
            
            if sample_data['metadatas']:
                for metadata in sample_data['metadatas']:
                    content_type = metadata.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    doc_name = metadata.get('document_name', 'unknown')
                    documents[doc_name] = documents.get(doc_name, 0) + 1
            
            return {
                'total_items': count,
                'content_types': content_types,
                'documents': documents,
                'collection_name': self.collection_name,
                'text_embedding_dim': self.text_embedding_dim,
                'image_embedding_dim': self.image_embedding_dim
            }
            
        except Exception as e:
            return {'error': f'Failed to get stats: {str(e)}'}