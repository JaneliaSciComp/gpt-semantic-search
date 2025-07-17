from typing import Optional
import io
import os
import tempfile
from .base import BasePDFScraper

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import convert_to_dict
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class UnstructuredScraper(BasePDFScraper):
    """Unstructured.io PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("unstructured")
        
        # Configuration options for unstructured
        self.strategy = "auto"  # Can be "fast", "hi_res", "auto"
        self.extract_images = False  # Set to True to extract images
        self.infer_table_structure = True  # Detect table structures
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using Unstructured.io"""
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured is not available. Install with: pip install unstructured[pdf]")
        
        try:
            # Partition the PDF into elements
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                extract_images_in_pdf=self.extract_images,
                infer_table_structure=self.infer_table_structure,
                chunking_strategy="by_title",  # Group content by titles
                max_characters=1000,  # Max chars per chunk
                combine_text_under_n_chars=500,  # Combine small chunks
            )
            
            # Convert elements to text
            text_content = []
            
            for element in elements:
                # Get element text and type
                element_text = str(element)
                element_type = element.category if hasattr(element, 'category') else 'unknown'
                
                # You can customize how different element types are handled
                if element_type in ['Title', 'Header']:
                    text_content.append(f"# {element_text}")
                elif element_type == 'Table':
                    text_content.append(f"**Table:**\n{element_text}")
                elif element_type == 'ListItem':
                    text_content.append(f"- {element_text}")
                else:
                    text_content.append(element_text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            print(f"Unstructured extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using Unstructured.io"""
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured is not available. Install with: pip install unstructured[pdf]")
        
        try:
            # Unstructured can work with file-like objects, but it's more reliable with temp files
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_file.flush()
                temp_path = temp_file.name
            
            # Extract using file path
            result = self.extract_text(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Unstructured extraction from bytes failed: {e}")
            return None
    
    def extract_with_structure(self, pdf_path: str) -> Optional[dict]:
        """Extract structured data including tables, images, and metadata"""
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured is not available. Install with: pip install unstructured[pdf]")
        
        try:
            # Partition with enhanced options
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",  # Use high-resolution strategy for better structure detection
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=1000,
                combine_text_under_n_chars=500,
            )
            
            # Convert to structured format
            structured_data = {
                'text_content': [],
                'tables': [],
                'images': [],
                'metadata': []
            }
            
            for element in elements:
                element_dict = convert_to_dict([element])[0]
                
                if element.category == 'Table':
                    structured_data['tables'].append({
                        'text': str(element),
                        'metadata': element_dict.get('metadata', {})
                    })
                elif element.category == 'Image':
                    structured_data['images'].append({
                        'text': str(element),
                        'metadata': element_dict.get('metadata', {})
                    })
                else:
                    structured_data['text_content'].append({
                        'text': str(element),
                        'type': element.category,
                        'metadata': element_dict.get('metadata', {})
                    })
            
            return structured_data
            
        except Exception as e:
            print(f"Unstructured structured extraction failed: {e}")
            return None