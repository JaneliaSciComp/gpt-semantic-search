from typing import Optional
from .base import BasePDFScraper

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
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
            result = self.converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using Docling"""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not available. Install with: pip install docling")
        
        try:
            result = self.converter.convert_bytes(pdf_bytes)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling extraction from bytes failed: {e}")
            return None