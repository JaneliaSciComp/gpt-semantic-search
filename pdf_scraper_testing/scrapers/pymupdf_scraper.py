from typing import Optional
import io
from .base import BasePDFScraper

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PyMuPDFScraper(BasePDFScraper):
    """PyMuPDF (fitz) PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("pymupdf")
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is not available. Install with: pip install PyMuPDF")
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is not available. Install with: pip install PyMuPDF")
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PyMuPDF extraction from bytes failed: {e}")
            return None