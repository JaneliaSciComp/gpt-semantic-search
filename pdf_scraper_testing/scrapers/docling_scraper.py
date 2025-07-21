from typing import Optional
from .base import BasePDFScraper
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import DocumentConversionInput
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False


class DoclingPDFScraper(BasePDFScraper):
    """Docling PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("docling")
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
        else:
            self.converter = None
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using Docling"""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not available. Install with: pip install docling")
        
        try:
            # Create proper DocumentConversionInput object
            input_obj = DocumentConversionInput.from_paths([Path(pdf_path)])
            result_generator = self.converter.convert(input_obj)
            # Get the first (and only) result from the generator  
            result = next(result_generator)
            # Use the render_as_markdown method
            return result.render_as_markdown()
        except Exception as e:
            import traceback
            print(f"Docling extraction failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using Docling"""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not available. Install with: pip install docling")
        
        try:
            # Create DocumentConversionInput from bytes
            input_obj = DocumentConversionInput.from_bytes(pdf_bytes)
            result_generator = self.converter.convert(input_obj)
            # Get the first (and only) result from the generator
            result = next(result_generator)
            return result.render_as_markdown()
        except Exception as e:
            print(f"Docling extraction from bytes failed: {e}")
            return None