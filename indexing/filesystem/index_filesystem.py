#!/usr/bin/env python

import argparse
import sys
import logging
import warnings
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional, Callable
from dataclasses import dataclass

from llama_index.core import Document
from indexing.weaviate_indexer import Indexer
from scraping.filesystem.file_processor import (
    process_file,
    is_supported_file_type,
    get_processing_method
)

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE = "Filesystem"

@dataclass
class ProcessingStats:
    """Statistics for file processing pipeline."""
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    batches_indexed: int = 0
    total_batches: int = 0

class StreamingFilesystemProcessor:
    """Stream-based filesystem processor for memory-efficient, progressive indexing."""
    
    def __init__(self, data_path, batch_size=20, max_workers=5, debug=False):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.debug = debug
        self.stats = ProcessingStats()
        
        if not self.data_path.exists():
            raise ValueError(f"Path does not exist: {self.data_path}")
        if not self.data_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.data_path}")
    
    def create_document(self, file_path, title, link, text):
        """Create a Document object following the existing pattern."""
        metadata = {
            "title": title,
            "link": link,
            "source": SOURCE,
            "file_path": str(file_path),
            "processing_method": get_processing_method(file_path)
        }
        return Document(text=text, doc_id=str(file_path), metadata=metadata)
    
    def is_hidden(self, path):
        """Check if a file or any parent directory is hidden."""
        return any(part.startswith('.') for part in path.parts)
    
    async def discover_files(self) -> AsyncGenerator[Path, None]:
        """Async generator that discovers files without loading them into memory."""
        logger.info(f"SEARCHING: Starting file discovery in: {self.data_path}")
        
        for file_path in self.data_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip hidden files
            if self.is_hidden(file_path):
                if self.debug:
                    logger.debug(f"SEARCHING: Skipping hidden file: {file_path}")
                self.stats.files_skipped += 1
                continue
            
            # Skip unsupported file types
            if not is_supported_file_type(file_path):
                if self.debug:
                    logger.debug(f"SEARCHING: Skipping unsupported file type: {file_path}")
                self.stats.files_skipped += 1
                continue
            
            self.stats.files_discovered += 1
            if self.stats.files_discovered % 100 == 0:
                logger.info(f"SEARCHING: Discovered {self.stats.files_discovered} supported files...")
            
            yield file_path
            
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
    
    async def process_file(self, file_path: Path) -> Optional[Document]:
        """Process a single file and return a Document."""
        try:
            # Process the file to extract text
            text_content = process_file(file_path)
            
            if not text_content or not text_content.strip():
                logger.warning(f"SCRAPING: No content extracted from: {file_path}")
                self.stats.files_failed += 1
                return None
            
            # Log first 20 chars being indexed for debugging
            preview = text_content[:20].replace('\n', '\\n').replace('\r', '\\r')
            if self.debug:
                logger.info(f"SCRAPING: {file_path.name}: First 20 chars = '{preview}'")
            
            # Create document metadata
            title = file_path.name
            link = f"file://{file_path.absolute()}"
            
            # Create and return document
            doc = self.create_document(file_path, title, link, text_content)
            self.stats.files_processed += 1
            
            return doc
            
        except Exception as e:
            logger.error(f"SCRAPING: Failed to process {file_path}: {e}")
            self.stats.files_failed += 1
            return None
    
    async def process_files_batch(
        self,
        files: list[Path],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> list[Document]:
        """Process a batch of files concurrently."""
        # Use semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_limit(file_path):
            async with semaphore:
                return await self.process_file(file_path)
        
        # Process files concurrently
        tasks = [process_with_limit(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"SCRAPING: Exception processing {files[i]}: {result}")
                self.stats.files_failed += 1
            elif result is not None:
                documents.append(result)
        
        # Update progress
        if progress_callback:
            progress_callback("scraping", self.stats.files_processed, self.stats.files_discovered)
        
        return documents
    
    async def stream_document_batches(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> AsyncGenerator[list[Document], None]:
        """Stream documents in batches for progressive indexing."""
        batch_files = []
        
        # Discover and process files in batches
        async for file_path in self.discover_files():
            batch_files.append(file_path)
            
            # Process batch when it reaches batch_size
            if len(batch_files) >= self.batch_size:
                logger.info(f"SCRAPING: Processing batch of {len(batch_files)} files...")
                
                documents = await self.process_files_batch(batch_files, progress_callback)
                if documents:
                    yield documents
                
                batch_files = []
        
        # Process remaining files in final batch
        if batch_files:
            logger.info(f"SCRAPING: Processing final batch of {len(batch_files)} files...")
            documents = await self.process_files_batch(batch_files, progress_callback)
            if documents:
                yield documents
        
        logger.info(f"SCRAPING: Complete - {self.stats.files_processed} files processed, {self.stats.files_failed} failed, {self.stats.files_skipped} skipped")


class FilesystemLoader:
    """Legacy loader - maintained for backward compatibility."""
    
    def __init__(self, data_path, debug=False):
        self.processor = StreamingFilesystemProcessor(data_path, debug=debug)
    
    def load_all_documents(self):
        """Legacy method - converts async streaming to sync batch loading."""
        async def _load_all():
            documents = []
            async for batch in self.processor.stream_document_batches():
                documents.extend(batch)
            return documents
        
        return asyncio.run(_load_all())


def main():
    parser = argparse.ArgumentParser(description='Load filesystem documents into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, 
                       help='Path to directory to index recursively')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", 
                       help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Janelia", 
                       help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-r', '--remove-existing', default=False, action=argparse.BooleanOptionalAction, 
                       help='Remove existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction, 
                       help='Print debugging information, such as file processing details.')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load documents from filesystem
    loader = FilesystemLoader(args.input, debug=args.debug)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    if not documents:
        logger.warning("No documents were loaded. Check your input directory and supported file types.")
        return

    # Index the documents in Weaviate
    indexer = Indexer(args.weaviate_url, args.class_prefix, args.remove_existing)
    indexer.index(documents)


if __name__ == '__main__':
    main()