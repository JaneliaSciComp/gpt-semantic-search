"""
Universal file scraper for comprehensive text extraction using Docling.

This module provides a unified interface for extracting text from various file types
using Docling for document processing and native reading for code/text files.
Docling provides superior document processing with AI-powered layout analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .file_processor import (
    process_file,
    get_supported_extensions,
    get_processing_method,
    is_supported_file_type,
    ALL_SUPPORTED_EXTENSIONS
)

logger = logging.getLogger(__name__)


class UniversalFileScraper:
    """
    Universal file scraper that can extract text from 45+ file types using Docling.
    
    Supports documents (PDF, Word, Excel, PowerPoint), code files (Python, Java, etc.),
    text files (Markdown, JSON, XML), Jupyter notebooks, and images with OCR via Docling.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the universal file scraper.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.supported_extensions = get_supported_extensions()
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"Initialized Docling-based scraper with {len(self.supported_extensions)} supported extensions")
    
    def get_supported_extensions(self) -> set:
        """Get all supported file extensions."""
        return self.supported_extensions.copy()
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if a file type is supported for scraping."""
        return is_supported_file_type(file_path)
    
    def get_file_info(self, file_path: Path) -> Dict[str, str]:
        """
        Get information about how a file will be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        return {
            'extension': file_path.suffix.lower(),
            'processing_method': get_processing_method(file_path),
            'supported': str(self.is_supported(file_path)),
            'size_bytes': str(file_path.stat().st_size) if file_path.exists() else 'unknown'
        }
    
    def scrape_file(self, file_path: Path) -> Tuple[str, Dict[str, any]]:
        """
        Scrape text content from a single file.
        
        Args:
            file_path: Path to the file to scrape
            
        Returns:
            Tuple of (text_content, metadata)
            
        Raises:
            ValueError: If file is not supported
            FileNotFoundError: If file doesn't exist
            RuntimeError: If scraping fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            # Extract text using the file processor
            text_content = process_file(file_path)
            
            # Log the exact scraped content for debugging
            logger.info(f"SCRAPED CONTENT from {file_path.name}:")
            logger.info(f"Full text content:\n{text_content}")
            logger.info(f"--- END SCRAPED CONTENT ---")
            
            # Create metadata
            metadata = {
                'file_path': str(file_path.absolute()),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'processing_method': get_processing_method(file_path),
                'extension': file_path.suffix.lower(),
                'character_count': len(text_content),
                'word_count': len(text_content.split()) if text_content else 0
            }
            
            logger.info(f"Scraped {metadata['character_count']} chars from {file_path.name}")
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Failed to scrape {file_path}: {e}")
            raise RuntimeError(f"Scraping failed for {file_path}: {e}")
    
    def scrape_directory(self, directory_path: Path, 
                        recursive: bool = True,
                        include_hidden: bool = False) -> List[Tuple[Path, str, Dict[str, any]]]:
        """
        Scrape all supported files from a directory.
        
        Args:
            directory_path: Path to directory to scrape
            recursive: Whether to scrape subdirectories
            include_hidden: Whether to include hidden files
            
        Returns:
            List of tuples: (file_path, text_content, metadata)
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        results = []
        processed_count = 0
        skipped_count = 0
        
        # Get file iterator based on recursive setting
        if recursive:
            file_iterator = directory_path.rglob("*")
        else:
            file_iterator = directory_path.iterdir()
        
        logger.info(f"Starting directory scrape: {directory_path}")
        
        for file_path in file_iterator:
            if not file_path.is_file():
                continue
            
            # Skip hidden files unless requested
            if not include_hidden and any(part.startswith('.') for part in file_path.parts):
                if self.debug:
                    logger.debug(f"Skipping hidden file: {file_path}")
                skipped_count += 1
                continue
            
            # Skip unsupported files
            if not self.is_supported(file_path):
                if self.debug:
                    logger.debug(f"Skipping unsupported file: {file_path}")
                skipped_count += 1
                continue
            
            try:
                text_content, metadata = self.scrape_file(file_path)
                results.append((file_path, text_content, metadata))
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Scraped {processed_count} files...")
                    
            except Exception as e:
                logger.error(f"Failed to scrape {file_path}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Directory scrape complete: {processed_count} processed, {skipped_count} skipped")
        return results
    
    def get_scraping_stats(self, results: List[Tuple[Path, str, Dict[str, any]]]) -> Dict[str, any]:
        """
        Get statistics about scraping results.
        
        Args:
            results: Results from scrape_directory
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {'total_files': 0}
        
        total_chars = sum(len(text) for _, text, _ in results)
        total_words = sum(metadata['word_count'] for _, _, metadata in results)
        total_size = sum(metadata['file_size'] for _, _, metadata in results)
        
        # Count by processing method
        method_counts = {}
        for _, _, metadata in results:
            method = metadata['processing_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Count by extension
        ext_counts = {}
        for file_path, _, _ in results:
            ext = file_path.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        return {
            'total_files': len(results),
            'total_characters': total_chars,
            'total_words': total_words,
            'total_file_size': total_size,
            'avg_chars_per_file': total_chars // len(results) if results else 0,
            'processing_methods': method_counts,
            'extensions': ext_counts
        }


def main():
    """Demo of universal file scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal File Scraper Demo')
    parser.add_argument('path', type=Path, help='File or directory path to scrape')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--include-hidden', action='store_true', help='Include hidden files')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive directory scraping')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = UniversalFileScraper(debug=args.debug)
    
    if args.path.is_file():
        # Scrape single file
        try:
            text_content, metadata = scraper.scrape_file(args.path)
            print(f"Scraped {args.path.name}:")
            print(f"  Characters: {metadata['character_count']:,}")
            print(f"  Words: {metadata['word_count']:,}")
            print(f"  Method: {metadata['processing_method']}")
            print(f"  Preview: {text_content[:200]}...")
        except Exception as e:
            print(f"Error scraping file: {e}")
    
    elif args.path.is_dir():
        # Scrape directory
        try:
            results = scraper.scrape_directory(
                args.path,
                recursive=not args.no_recursive,
                include_hidden=args.include_hidden
            )
            
            stats = scraper.get_scraping_stats(results)
            print(f"\nScraping Statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Total words: {stats['total_words']:,}")
            print(f"  Processing methods: {stats['processing_methods']}")
            
        except Exception as e:
            print(f"Error scraping directory: {e}")
    
    else:
        print(f"Path does not exist: {args.path}")


if __name__ == '__main__':
    main()