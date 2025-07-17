from typing import Optional
import io
from .base import BasePDFScraper

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


class PDFMinerScraper(BasePDFScraper):
    """PDFMiner PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("pdfminer")
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PDFMiner"""
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer is not available. Install with: pip install pdfminer.six")
        
        try:
            text = extract_text(pdf_path)
            return text.strip()
        except Exception as e:
            print(f"PDFMiner extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using PDFMiner"""
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer is not available. Install with: pip install pdfminer.six")
        
        try:
            text = extract_text(io.BytesIO(pdf_bytes))
            return text.strip()
        except Exception as e:
            print(f"PDFMiner extraction from bytes failed: {e}")
            return None