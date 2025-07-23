"""
Filesystem monitoring and scraping module for comprehensive file processing using Docling.

This module provides:
1. High-performance filesystem monitoring using watchfiles for real-time updates
2. Universal file scraping for 45+ file types using Docling AI-powered processing and native methods
3. Comprehensive file processing capabilities for RAG applications with superior accuracy
"""

from .filesystem_monitor import (
    FilesystemMonitor,
    watch_directory,
    process_file,
    remove_file,
    is_supported_extension,
    should_process_file,
    SUPPORTED_EXTENSIONS,
)

from .file_processor import (
    process_file as process_file_sync,
    get_supported_extensions,
    get_processing_method,
    is_supported_file_type,
    ALL_SUPPORTED_EXTENSIONS,
    DOCLING_EXTENSIONS,
    CODE_EXTENSIONS,
    TEXT_EXTENSIONS,
    JUPYTER_EXTENSIONS,
)

from .universal_scraper import (
    UniversalFileScraper,
)

__all__ = [
    # Filesystem monitoring
    "FilesystemMonitor",
    "watch_directory", 
    "process_file",
    "remove_file",
    "is_supported_extension",
    "should_process_file",
    "SUPPORTED_EXTENSIONS",
    
    # File processing
    "process_file_sync",
    "get_supported_extensions",
    "get_processing_method",
    "is_supported_file_type",
    "ALL_SUPPORTED_EXTENSIONS",
    "DOCLING_EXTENSIONS",
    "CODE_EXTENSIONS", 
    "TEXT_EXTENSIONS",
    "JUPYTER_EXTENSIONS",
    
    # Universal scraping
    "UniversalFileScraper",
]