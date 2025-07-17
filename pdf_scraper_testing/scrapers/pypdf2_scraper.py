from typing import Optional
import io
from .base import BasePDFScraper

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class PyPDF2Scraper(BasePDFScraper):
    """PyPDF2 PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("pypdf2")
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PyPDF2"""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is not available. Install with: pip install PyPDF2")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using PyPDF2"""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is not available. Install with: pip install PyPDF2")
        
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction from bytes failed: {e}")
            return None