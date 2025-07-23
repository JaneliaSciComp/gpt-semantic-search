"""
Enhanced file processing module using Docling for comprehensive text extraction.

This module provides advanced text extraction capabilities optimized for RAG applications.
It uses Docling for document processing and native reading for code files and simple text formats.
Docling offers superior document processing with AI-powered layout analysis and OCR capabilities.
"""

import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Set, Optional, Any
import chardet

# Configure logging
logger = logging.getLogger(__name__)

# Supported file extensions organized by processing method
DOCLING_EXTENSIONS: Set[str] = {
    # Documents (Docling's core strength)
    '.pdf', '.docx', '.xlsx', '.pptx',
    # Web and markup formats
    '.html', '.htm', '.md', '.adoc',
    # Data formats
    '.csv', '.json',
    # Images with OCR capabilities
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'
}

CODE_EXTENSIONS: Set[str] = {
    # Programming languages
    '.py', '.java', '.jl', '.cpp', '.c', '.h', '.hpp',
    '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.rb',
    '.php', '.swift', '.kt', '.scala', '.clj', '.hs',
    '.r', '.m', '.sh', '.bash', '.ps1', '.bat',
    # Configuration and data
    '.xml', '.yaml', '.yml', '.toml', '.ini',
    '.cfg', '.conf', '.properties', '.env'
}

TEXT_EXTENSIONS: Set[str] = {
    # Plain text and documentation
    '.txt', '.rst', '.tex', '.org',
    # Data files
    '.tsv', '.log',
    # Markup
    '.xml', '.svg'
}

JUPYTER_EXTENSIONS: Set[str] = {
    '.ipynb'
}

# All supported extensions
ALL_SUPPORTED_EXTENSIONS: Set[str] = (
    DOCLING_EXTENSIONS | CODE_EXTENSIONS | TEXT_EXTENSIONS | JUPYTER_EXTENSIONS
)

# MIME type to extension mapping for docling-supported formats
MIME_TO_EXTENSION: Dict[str, str] = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'text/html': '.html',
    'text/markdown': '.md',
    'text/csv': '.csv',
    'application/json': '.json',
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/tiff': '.tiff',
    'image/bmp': '.bmp',
    'image/webp': '.webp'
}


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding using chardet.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding or 'utf-8' as fallback
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
            
            # Fall back to utf-8 if confidence is too low
            if confidence < 0.7:
                logger.warning(f"Low confidence encoding detection for {file_path}, using utf-8")
                return 'utf-8'
                
            return encoding or 'utf-8'
    except Exception as e:
        logger.warning(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'


def get_file_type_by_mime(file_path: Path) -> Optional[str]:
    """
    Determine file type using MIME type detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension or None if not supported
    """
    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return MIME_TO_EXTENSION.get(mime_type)
    except Exception as e:
        logger.debug(f"MIME type detection failed for {file_path}: {e}")
    return None


def is_supported_file_type(file_path: Path) -> bool:
    """
    Check if a file type is supported for processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file type is supported
    """
    # Check by extension first
    extension = file_path.suffix.lower()
    if extension in ALL_SUPPORTED_EXTENSIONS:
        return True
    
    # Check by MIME type for files without clear extensions
    mime_extension = get_file_type_by_mime(file_path)
    if mime_extension and mime_extension in ALL_SUPPORTED_EXTENSIONS:
        return True
    
    return False


def get_processing_method(file_path: Path) -> str:
    """
    Determine the processing method for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Processing method name
    """
    extension = file_path.suffix.lower()
    
    if extension in DOCLING_EXTENSIONS:
        return "docling"
    elif extension in CODE_EXTENSIONS:
        return "native_code"
    elif extension in TEXT_EXTENSIONS:
        return "native_text"
    elif extension in JUPYTER_EXTENSIONS:
        return "jupyter"
    else:
        # Check MIME type
        mime_extension = get_file_type_by_mime(file_path)
        if mime_extension in DOCLING_EXTENSIONS:
            return "docling"
        return "unknown"


def get_supported_extensions() -> Set[str]:
    """
    Get all supported file extensions.
    
    Returns:
        Set of supported file extensions
    """
    return ALL_SUPPORTED_EXTENSIONS.copy()


def process_docling_file(file_path: Path) -> str:
    """
    Process a file using Docling for text extraction.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Extracted text content
        
    Raises:
        ImportError: If docling is not available
        RuntimeError: If processing fails
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.document import DocumentConversionInput
        
        logger.debug(f"Processing with Docling: {file_path}")
        
        # Initialize Docling converter
        converter = DocumentConverter()
        
        # Create proper DocumentConversionInput object
        input_obj = DocumentConversionInput.from_paths([file_path])
        result_generator = converter.convert(input_obj)
        
        # Get the first (and only) result from the generator  
        result = next(result_generator)
        
        # Use the render_as_markdown method
        text_content = result.render_as_markdown()
        
        if not text_content or not text_content.strip():
            logger.warning(f"Empty content extracted by Docling from {file_path}")
            return ""
        
        logger.debug(f"Docling extracted {len(text_content)} characters from {file_path}")
        return text_content.strip()
        
    except ImportError:
        raise ImportError("Docling library is required for processing this file type. Install with: pip install docling")
    except Exception as e:
        logger.error(f"Docling processing failed for {file_path}: {e}")
        raise RuntimeError(f"Failed to process {file_path} with Docling: {e}")


def process_native_text_file(file_path: Path) -> str:
    """
    Process plain text files using native Python reading.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
        
    Raises:
        RuntimeError: If reading fails
    """
    try:
        encoding = detect_encoding(file_path)
        logger.debug(f"Reading text file with encoding {encoding}: {file_path}")
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"Empty content in text file: {file_path}")
            return ""
        
        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content.strip()
        
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        raise RuntimeError(f"Failed to read text file {file_path}: {e}")


def process_native_code_file(file_path: Path) -> str:
    """
    Process code files using native Python reading with syntax awareness.
    
    Args:
        file_path: Path to the code file
        
    Returns:
        File content as string with metadata
        
    Raises:
        RuntimeError: If reading fails
    """
    try:
        encoding = detect_encoding(file_path)
        logger.debug(f"Reading code file with encoding {encoding}: {file_path}")
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"Empty content in code file: {file_path}")
            return ""
        
        # Add file type context for better RAG performance
        file_type = file_path.suffix.lower().lstrip('.')
        enhanced_content = f"File: {file_path.name} (Type: {file_type})\n\n{content}"
        
        logger.debug(f"Read {len(content)} characters from code file {file_path}")
        return enhanced_content.strip()
        
    except Exception as e:
        logger.error(f"Failed to read code file {file_path}: {e}")
        raise RuntimeError(f"Failed to read code file {file_path}: {e}")


def process_jupyter_notebook(file_path: Path) -> str:
    """
    Process Jupyter notebook files by extracting code and markdown cells.
    
    Args:
        file_path: Path to the .ipynb file
        
    Returns:
        Combined content from all cells
        
    Raises:
        RuntimeError: If processing fails
    """
    try:
        logger.debug(f"Processing Jupyter notebook: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        content_parts = [f"Jupyter Notebook: {file_path.name}\n"]
        
        cells = notebook.get('cells', [])
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', 'unknown')
            source = cell.get('source', [])
            
            if isinstance(source, list):
                cell_content = ''.join(source)
            else:
                cell_content = str(source)
            
            if cell_content.strip():
                content_parts.append(f"\n--- Cell {i+1} ({cell_type}) ---\n{cell_content}")
        
        combined_content = '\n'.join(content_parts)
        
        if not combined_content.strip():
            logger.warning(f"No content extracted from Jupyter notebook: {file_path}")
            return ""
        
        logger.debug(f"Extracted {len(combined_content)} characters from Jupyter notebook {file_path}")
        return combined_content.strip()
        
    except Exception as e:
        logger.error(f"Failed to process Jupyter notebook {file_path}: {e}")
        raise RuntimeError(f"Failed to process Jupyter notebook {file_path}: {e}")


def process_file(file_path: Path) -> str:
    """
    Process a file and extract text content using the appropriate method.
    
    This is the main entry point for file processing that determines the best
    processing method based on file type and delegates to specialized processors.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
        RuntimeError: If processing fails
    """
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    if not is_supported_file_type(file_path):
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    processing_method = get_processing_method(file_path)
    logger.debug(f"Processing {file_path} using method: {processing_method}")
    
    try:
        if processing_method == "docling":
            return process_docling_file(file_path)
        elif processing_method == "native_text":
            return process_native_text_file(file_path)
        elif processing_method == "native_code":
            return process_native_code_file(file_path)
        elif processing_method == "jupyter":
            return process_jupyter_notebook(file_path)
        else:
            raise ValueError(f"Unknown processing method: {processing_method}")
            
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        raise


def get_file_stats() -> Dict[str, Any]:
    """
    Get statistics about supported file types and processing methods.
    
    Returns:
        Dictionary with file processing statistics
    """
    return {
        'total_supported_extensions': len(ALL_SUPPORTED_EXTENSIONS),
        'docling_extensions': len(DOCLING_EXTENSIONS),
        'code_extensions': len(CODE_EXTENSIONS),
        'text_extensions': len(TEXT_EXTENSIONS),
        'jupyter_extensions': len(JUPYTER_EXTENSIONS),
        'processing_methods': {
            'docling': list(DOCLING_EXTENSIONS),
            'native_code': list(CODE_EXTENSIONS),
            'native_text': list(TEXT_EXTENSIONS),
            'jupyter': list(JUPYTER_EXTENSIONS)
        }
    }


if __name__ == "__main__":
    # Demo/test functionality
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python file_processor.py <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    try:
        print(f"Processing: {file_path}")
        print(f"Supported: {is_supported_file_type(file_path)}")
        print(f"Method: {get_processing_method(file_path)}")
        
        if is_supported_file_type(file_path):
            content = process_file(file_path)
            print(f"Content length: {len(content)} characters")
            print(f"Preview: {content[:200]}...")
        else:
            print("File type not supported")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)