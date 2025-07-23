"""
Enhanced file processing module for comprehensive text extraction.

This module provides advanced text extraction capabilities for a wide range of file types,
optimized for RAG applications. It uses unstructured for complex documents and native
reading for code files and simple text formats.
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
UNSTRUCTURED_EXTENSIONS: Set[str] = {
    # Documents
    '.pdf', '.docx', '.doc', '.pptx', '.ppt',
    # Web and email
    '.html', '.htm', '.eml',
    # Images with text (requires OCR capabilities)
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp'
}

CODE_EXTENSIONS: Set[str] = {
    # Programming languages
    '.py', '.java', '.jl', '.cpp', '.c', '.h', '.hpp',
    '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.rb',
    '.php', '.swift', '.kt', '.scala', '.clj', '.hs',
    '.r', '.m', '.sh', '.bash', '.ps1', '.bat',
    # Configuration and data
    '.json', '.xml', '.yaml', '.yml', '.toml', '.ini',
    '.cfg', '.conf', '.properties', '.env'
}

TEXT_EXTENSIONS: Set[str] = {
    # Plain text and documentation
    '.txt', '.md', '.rst', '.tex', '.org',
    # Data files
    '.csv', '.tsv', '.log',
    # Markup
    '.xml', '.svg'
}

JUPYTER_EXTENSIONS: Set[str] = {
    '.ipynb'
}

# Combined set of all supported extensions
ALL_SUPPORTED_EXTENSIONS: Set[str] = (
    UNSTRUCTURED_EXTENSIONS | CODE_EXTENSIONS | TEXT_EXTENSIONS | JUPYTER_EXTENSIONS
)

# MIME type mappings for fallback detection
MIME_TO_EXTENSION: Dict[str, str] = {
    'application/pdf': '.pdf',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'text/html': '.html',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'application/json': '.json',
    'application/xml': '.xml',
    'text/xml': '.xml',
    'text/csv': '.csv',
    'application/x-python': '.py',
    'text/x-python': '.py',
    'application/javascript': '.js',
    'text/javascript': '.js',
}


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding using chardet.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding string
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
    
    # Fallback to MIME type detection
    mime_extension = get_file_type_by_mime(file_path)
    return mime_extension is not None


def process_with_unstructured(file_path: Path) -> str:
    """
    Process complex documents using unstructured library.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    try:
        from unstructured.partition.auto import partition
        
        # Partition the document
        elements = partition(filename=str(file_path))
        
        # Extract text from elements and combine
        text_content = []
        for element in elements:
            if hasattr(element, 'text') and element.text.strip():
                text_content.append(element.text.strip())
        
        # Join with double newlines to preserve semantic separation
        combined_text = '\n\n'.join(text_content)
        
        logger.info(f"Extracted {len(combined_text)} characters from {file_path.name} using unstructured")
        return combined_text
        
    except ImportError:
        raise ImportError("unstructured library is required for processing this file type")
    except Exception as e:
        raise RuntimeError(f"Failed to process {file_path} with unstructured: {e}")


def process_jupyter_notebook(file_path: Path) -> str:
    """
    Process Jupyter notebook files by extracting code and markdown cells.
    
    Args:
        file_path: Path to the .ipynb file
        
    Returns:
        Extracted text content from cells
    """
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            notebook = json.load(f)
        
        text_content = []
        
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            if isinstance(source, list):
                cell_content = ''.join(source)
            else:
                cell_content = str(source)
            
            if cell_content.strip():
                if cell_type == 'markdown':
                    text_content.append(f"# Markdown Cell\n{cell_content}")
                elif cell_type == 'code':
                    text_content.append(f"# Code Cell\n{cell_content}")
                else:
                    text_content.append(cell_content)
        
        combined_text = '\n\n'.join(text_content)
        logger.info(f"Extracted {len(combined_text)} characters from Jupyter notebook {file_path.name}")
        return combined_text
        
    except Exception as e:
        raise RuntimeError(f"Failed to process Jupyter notebook {file_path}: {e}")


def process_text_file(file_path: Path) -> str:
    """
    Process plain text, code, and structured data files.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string
    """
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        logger.info(f"Read {len(content)} characters from {file_path.name}")
        return content
        
    except UnicodeDecodeError as e:
        logger.warning(f"Unicode decode error for {file_path}, trying latin-1: {e}")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            logger.info(f"Successfully read {len(content)} characters from {file_path.name} using latin-1")
            return content
        except Exception as fallback_e:
            raise RuntimeError(f"Failed to read {file_path} with multiple encodings: {fallback_e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read text file {file_path}: {e}")


def process_file(file_path: Path) -> str:
    """
    Process a file and extract its text content based on file type.
    
    This function determines the file type by extension or MIME type and uses the
    appropriate processing method:
    - Unstructured library for complex documents (PDFs, Office docs, HTML, etc.)
    - Native text reading for code files and plain text
    - Special handling for Jupyter notebooks
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Extracted text content as string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If processing fails
        ValueError: If file type is not supported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    # Get file extension
    extension = file_path.suffix.lower()
    
    # If extension is not recognized, try MIME type detection
    if extension not in ALL_SUPPORTED_EXTENSIONS:
        mime_extension = get_file_type_by_mime(file_path)
        if mime_extension:
            extension = mime_extension
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported extensions: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )
    
    logger.info(f"Processing {file_path.name} as {extension} file")
    
    try:
        # Route to appropriate processor
        if extension in JUPYTER_EXTENSIONS:
            return process_jupyter_notebook(file_path)
        elif extension in UNSTRUCTURED_EXTENSIONS:
            return process_with_unstructured(file_path)
        elif extension in CODE_EXTENSIONS or extension in TEXT_EXTENSIONS:
            return process_text_file(file_path)
        else:
            # This shouldn't happen if our logic is correct
            raise ValueError(f"No processor available for extension: {extension}")
            
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        raise


def get_supported_extensions() -> Set[str]:
    """
    Get the complete set of supported file extensions.
    
    Returns:
        Set of supported file extensions
    """
    return ALL_SUPPORTED_EXTENSIONS.copy()


def get_processing_method(file_path: Path) -> str:
    """
    Determine which processing method will be used for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Processing method name ('unstructured', 'jupyter', 'text', or 'unsupported')
    """
    extension = file_path.suffix.lower()
    
    if extension not in ALL_SUPPORTED_EXTENSIONS:
        mime_extension = get_file_type_by_mime(file_path)
        if mime_extension:
            extension = mime_extension
        else:
            return 'unsupported'
    
    if extension in JUPYTER_EXTENSIONS:
        return 'jupyter'
    elif extension in UNSTRUCTURED_EXTENSIONS:
        return 'unstructured'
    elif extension in CODE_EXTENSIONS or extension in TEXT_EXTENSIONS:
        return 'text'
    else:
        return 'unsupported'


# Utility function for filesystem monitor integration
def create_process_file_callback(custom_processor: Optional[Any] = None):
    """
    Create a process_file callback function that can be used with the filesystem monitor.
    
    Args:
        custom_processor: Optional custom processor function
        
    Returns:
        Async callback function
    """
    async def async_process_file(path: Path) -> str:
        """Async wrapper for process_file function."""
        if custom_processor:
            return await custom_processor(path)
        else:
            return process_file(path)
    
    return async_process_file