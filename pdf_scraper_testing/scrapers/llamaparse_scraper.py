from typing import Optional
import os
from .base import BasePDFScraper

try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False


class LlamaParseScraper(BasePDFScraper):
    """LlamaParse PDF scraper implementation"""
    
    def __init__(self):
        super().__init__("llamaparse")
        if LLAMAPARSE_AVAILABLE:
            # LlamaParse requires an API key
            api_key = os.getenv("LLAMA_CLOUD_API_KEY")
            if not api_key:
                raise ValueError("LLAMA_CLOUD_API_KEY environment variable is required for LlamaParse")
            
            self.parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",  # Can be "markdown" or "text"
                verbose=True,
                language="en",  # Optionally specify language
            )
        else:
            self.parser = None
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using LlamaParse"""
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("LlamaParse is not available. Install with: pip install llama-parse")
        
        if not self.parser:
            raise ValueError("LlamaParse parser not initialized. Check LLAMA_CLOUD_API_KEY.")
        
        try:
            # LlamaParse works with file paths
            documents = self.parser.load_data(pdf_path)
            
            # Combine all document pages/chunks
            text = ""
            for doc in documents:
                text += doc.text + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"LlamaParse extraction failed: {e}")
            return None
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using LlamaParse"""
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("LlamaParse is not available. Install with: pip install llama-parse")
        
        # LlamaParse doesn't directly support bytes, so we need to write to temp file
        import tempfile
        import os
        
        try:
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
            print(f"LlamaParse extraction from bytes failed: {e}")
            return None