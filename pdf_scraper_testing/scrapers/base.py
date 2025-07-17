from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from pathlib import Path


class BasePDFScraper(ABC):
    """Base class for PDF scrapers"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file"""
        pass
    
    @abstractmethod
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes"""
        pass
    
    def extract_with_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text with performance metadata"""
        start_time = time.time()
        
        try:
            text = self.extract_text(pdf_path)
            success = text is not None
            error = None
        except Exception as e:
            text = None
            success = False
            error = str(e)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get file size for performance metrics
        file_size = Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0
        
        return {
            'scraper': self.name,
            'text': text,
            'success': success,
            'error': error,
            'processing_time': processing_time,
            'file_size': file_size,
            'characters_extracted': len(text) if text else 0,
            'words_extracted': len(text.split()) if text else 0,
        }
    
    def extract_bytes_with_metadata(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text from bytes with performance metadata"""
        start_time = time.time()
        
        try:
            text = self.extract_text_from_bytes(pdf_bytes)
            success = text is not None
            error = None
        except Exception as e:
            text = None
            success = False
            error = str(e)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'scraper': self.name,
            'text': text,
            'success': success,
            'error': error,
            'processing_time': processing_time,
            'file_size': len(pdf_bytes),
            'characters_extracted': len(text) if text else 0,
            'words_extracted': len(text.split()) if text else 0,
        }