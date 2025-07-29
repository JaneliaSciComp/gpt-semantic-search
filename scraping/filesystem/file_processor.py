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
    # Documents (Docling's core strength - complex binary formats)
    '.pdf', '.docx', '.xlsx', '.pptx',
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
    '.cfg', '.conf', '.properties', '.env',
    # Web markup and structured data (moved from Docling)
    '.html', '.htm', '.csv', '.json'
}

TEXT_EXTENSIONS: Set[str] = {
    # Plain text and documentation
    '.txt', '.rst', '.tex', '.org',
    # Markdown and documentation formats (moved from Docling)
    '.md', '.adoc',
    # Data files
    '.tsv', '.log',
    # Markup (XML already in CODE_EXTENSIONS)
    '.svg'
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


def validate_file_content(file_path: Path) -> bool:
    """
    Validate that file content matches its extension to detect corrupted or misnamed files.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        True if file appears valid, False if suspicious or corrupted
    """
    try:
        extension = file_path.suffix.lower()
        
        # Check file size - skip empty files or suspiciously large text files
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.warning(f"VALIDATION: Skipping empty file: {file_path.name}")
                return False
            
            # Skip extremely large files that might cause processing issues
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"VALIDATION: Skipping large file ({file_size / 1024 / 1024:.1f}MB): {file_path.name}")
                return False
                
        except OSError as e:
            logger.warning(f"VALIDATION: Cannot access file stats for {file_path.name}: {e}")
            return False
        
        # Basic content validation for text-based files
        if extension in ['.md', '.txt', '.html', '.htm', '.json', '.csv', '.xml']:
            try:
                # Try to read first few bytes to validate it's text
                with open(file_path, 'rb') as f:
                    sample = f.read(1024)
                
                # Check for binary content in text files
                if b'\x00' in sample:  # Null bytes indicate binary content
                    logger.warning(f"VALIDATION: Binary content detected in text file: {file_path.name}")
                    return False
                
                # Check for valid JSON structure
                if extension == '.json':
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            import json
                            json.loads(f.read(10000))  # Try parsing first 10KB
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.warning(f"VALIDATION: Invalid JSON format: {file_path.name}")
                        return False
            
            except (OSError, IOError) as e:
                logger.warning(f"VALIDATION: Cannot read file for validation {file_path.name}: {e}")
                return False
        
        # Basic validation for image files
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(20)
                
                # Check for common image file signatures
                image_signatures = {
                    '.png': [b'\x89PNG\r\n\x1a\n'],
                    '.jpg': [b'\xff\xd8\xff'], '.jpeg': [b'\xff\xd8\xff'],
                    '.pdf': [b'%PDF'],
                    '.bmp': [b'BM'],
                    '.tiff': [b'II*\x00', b'MM\x00*'],
                    '.webp': [b'RIFF']
                }
                
                expected_sigs = image_signatures.get(extension, [])
                if expected_sigs and not any(header.startswith(sig) for sig in expected_sigs):
                    logger.warning(f"VALIDATION: File signature doesn't match extension for {file_path.name}")
                    return False
                    
            except (OSError, IOError) as e:
                logger.warning(f"VALIDATION: Cannot read file header for validation {file_path.name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"VALIDATION: File validation failed for {file_path.name}: {e}")
        return True  # Default to True to avoid blocking valid files


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
        from docling.datamodel.pipeline_options import PipelineOptions
        
        extension = file_path.suffix.lower()
        logger.info(f"SCRAPING: Starting Docling processing for {extension} file: {file_path.name}")
        
        # Configure Docling with optimized settings based on file type
        pipeline_options = PipelineOptions()
        
        # Optimize for different file types
        if extension in ['.pdf', '.docx', '.xlsx', '.pptx']:
            # Enable full processing for complex documents
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = False  # Disable OCR for faster processing of digital documents
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            # Enable OCR for images but disable table structure analysis
            pipeline_options.do_table_structure = False
            pipeline_options.do_ocr = True
        else:
            # Minimal processing for other file types
            pipeline_options.do_table_structure = False
            pipeline_options.do_ocr = False
        
        # Initialize Docling converter with optimized configuration
        converter = DocumentConverter(pipeline_options=pipeline_options)
        
        # Create proper DocumentConversionInput object
        input_obj = DocumentConversionInput.from_paths([file_path])
        result_generator = converter.convert(input_obj)
        
        # Get the first (and only) result from the generator  
        result = next(result_generator)
        
        # Use the render_as_markdown method
        text_content = result.render_as_markdown()
        
        if not text_content or not text_content.strip():
            logger.warning(f"SCRAPING: Empty content extracted by Docling from {file_path}")
            raise RuntimeError(f"Empty content extracted by Docling from {file_path.name}")
        
        # Log first 20 characters for debugging (matching existing log pattern)
        preview = text_content.strip()[:20].replace('\n', '\\n')
        logger.info(f"SCRAPING: {file_path.name}: First 20 chars = '{preview}'")
        logger.info(f"SCRAPING: Successfully extracted {len(text_content)} characters using Docling")
        
        return text_content.strip()
        
    except ImportError:
        logger.error(f"SCRAPING: Docling library not available for {file_path}")
        raise ImportError("Docling library is required for processing this file type. Install with: pip install docling")
    except Exception as e:
        # Provide more specific error information
        error_type = type(e).__name__
        if "pdf" in str(e).lower() or "pdfium" in str(e).lower():
            logger.error(f"SCRAPING: PDF backend error processing {file_path.name}: {error_type} - {e}")
            raise RuntimeError(f"PDF processing error for {file_path.name}: File may be corrupted or not a valid {extension} file")
        elif "data format" in str(e).lower():
            logger.error(f"SCRAPING: Data format error processing {file_path.name}: {error_type} - {e}")
            raise RuntimeError(f"Data format error for {file_path.name}: File format not supported by Docling backend")
        else:
            logger.error(f"SCRAPING: General Docling error processing {file_path.name}: {error_type} - {e}")
            raise RuntimeError(f"Docling processing failed for {file_path.name}: {e}")


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
        logger.info(f"SCRAPING: Reading text file {file_path.name} with encoding {encoding}")
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"SCRAPING: Empty content in text file: {file_path}")
            return ""
        
        # Log first 20 characters for debugging (matching existing log pattern)
        preview = content.strip()[:20].replace('\n', '\\n')
        logger.info(f"SCRAPING: {file_path.name}: First 20 chars = '{preview}'")
        logger.info(f"SCRAPING: Successfully read {len(content)} characters using native text processing")
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"SCRAPING: Failed to read text file {file_path.name}: {e}")
        raise RuntimeError(f"Failed to read text file {file_path.name}: {e}")


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
        logger.info(f"SCRAPING: Reading code file {file_path.name} with encoding {encoding}")
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"SCRAPING: Empty content in code file: {file_path}")
            return ""
        
        # Add file type context for better RAG performance
        file_type = file_path.suffix.lower().lstrip('.')
        enhanced_content = f"File: {file_path.name} (Type: {file_type})\n\n{content}"
        
        # Log first 20 characters for debugging (matching existing log pattern)
        preview = enhanced_content.strip()[:20].replace('\n', '\\n')
        logger.info(f"SCRAPING: {file_path.name}: First 20 chars = '{preview}'")
        logger.info(f"SCRAPING: Successfully read {len(content)} characters using native code processing")
        
        return enhanced_content.strip()
        
    except Exception as e:
        logger.error(f"SCRAPING: Failed to read code file {file_path.name}: {e}")
        raise RuntimeError(f"Failed to read code file {file_path.name}: {e}")


def process_native_office_file(file_path: Path) -> str:
    """
    Process Office documents using native Python processing as fallback.
    
    Args:
        file_path: Path to the Office document
        
    Returns:
        Extracted text content
        
    Raises:
        RuntimeError: If processing fails
    """
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from office_processor import OfficeProcessor
        
        extension = file_path.suffix.lower()
        logger.info(f"SCRAPING: Starting native Office processing for {extension} file: {file_path.name}")
        
        # Use the native Office processor
        content = OfficeProcessor.process_office_document(file_path)
        
        if not content or not content.strip():
            logger.warning(f"SCRAPING: Empty content extracted by native processing from {file_path}")
            raise RuntimeError(f"No content extracted from Office document: {file_path.name}")
        
        # Log first 20 characters for debugging (matching existing log pattern)
        preview = content.strip()[:20].replace('\n', '\\n')
        logger.info(f"SCRAPING: {file_path.name}: First 20 chars = '{preview}'")
        logger.info(f"SCRAPING: Successfully extracted {len(content)} characters using native Office processing")
        
        return content.strip()
        
    except ImportError as e:
        logger.error(f"SCRAPING: Office processor not available for {file_path}: {e}")
        raise RuntimeError(f"Native Office processor not available: {e}")
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"SCRAPING: Native Office processing error for {file_path.name}: {error_type} - {e}")
        raise RuntimeError(f"Native Office processing failed for {file_path.name}: {e}")


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
    Includes fallback processing for files that fail with primary method.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
        RuntimeError: If processing fails with all attempted methods
    """
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    if not is_supported_file_type(file_path):
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    # Validate file content before processing
    if not validate_file_content(file_path):
        raise ValueError(f"File validation failed - file may be corrupted or misnamed: {file_path.name}")
    
    processing_method = get_processing_method(file_path)
    extension = file_path.suffix.lower()
    logger.info(f"Processing {file_path} using method: {processing_method}")
    
    # Track attempted methods for error reporting
    attempted_methods = []
    last_error = None
    
    try:
        # Try primary processing method
        attempted_methods.append(processing_method)
        
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
            
    except Exception as primary_error:
        last_error = primary_error
        logger.warning(f"Primary processing method '{processing_method}' failed for {file_path}: {primary_error}")
        
        # Try fallback processing for certain file types
        if processing_method == "docling":
            # For files that failed Docling processing, try alternative methods
            fallback_method = None
            
            # Office documents that can be processed natively
            if extension in ['.docx', '.pptx', '.xlsx']:
                fallback_method = "native_office"
            # Text-like files that could be read natively
            elif extension in ['.md', '.adoc', '.html', '.htm']:
                fallback_method = "native_text"
            # Structured data files that could be read as code
            elif extension in ['.json', '.csv']:
                fallback_method = "native_code"
            
            if fallback_method:
                try:
                    attempted_methods.append(fallback_method)
                    logger.info(f"Attempting fallback processing for {file_path} using method: {fallback_method}")
                    
                    if fallback_method == "native_office":
                        return process_native_office_file(file_path)
                    elif fallback_method == "native_text":
                        return process_native_text_file(file_path)
                    elif fallback_method == "native_code":
                        return process_native_code_file(file_path)
                        
                except Exception as fallback_error:
                    last_error = fallback_error
                    logger.warning(f"Fallback processing method '{fallback_method}' also failed for {file_path}: {fallback_error}")
        
        # If we reach here, all processing attempts failed
        methods_tried = " -> ".join(attempted_methods)
        logger.error(f"All processing methods failed for {file_path}. Attempted: {methods_tried}. Final error: {last_error}")
        raise RuntimeError(f"Failed to process {file_path} with methods: {methods_tried}. Final error: {last_error}")


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