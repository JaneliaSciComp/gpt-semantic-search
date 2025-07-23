#!/usr/bin/env python

import argparse
import sys
import logging
import warnings
from pathlib import Path

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


class FilesystemLoader:
    """Load documents from filesystem using comprehensive file processing."""
    
    def __init__(self, data_path, debug=False):
        self.data_path = Path(data_path)
        self.debug = debug
        
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
        return Document(text=text, metadata=metadata)
    
    def is_hidden(self, path):
        """Check if a file or any parent directory is hidden."""
        return any(part.startswith('.') for part in path.parts)
    
    def load_all_documents(self):
        """Load all supported documents from the filesystem."""
        documents = []
        processed_count = 0
        skipped_count = 0
        
        logger.info(f"Starting filesystem indexing from: {self.data_path}")
        
        # Walk through all files recursively
        for file_path in self.data_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip hidden files
            if self.is_hidden(file_path):
                if self.debug:
                    logger.debug(f"Skipping hidden file: {file_path}")
                skipped_count += 1
                continue
            
            # Skip unsupported file types
            if not is_supported_file_type(file_path):
                if self.debug:
                    logger.debug(f"Skipping unsupported file type: {file_path}")
                skipped_count += 1
                continue
            
            try:
                # Process the file to extract text
                text_content = process_file(file_path)
                
                if not text_content or not text_content.strip():
                    logger.warning(f"No content extracted from: {file_path}")
                    skipped_count += 1
                    continue
                
                # Create document metadata
                title = file_path.name
                link = f"file://{file_path.absolute()}"
                
                # Create and add document
                doc = self.create_document(file_path, title, link, text_content)
                documents.append(doc)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} files...")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Filesystem indexing complete: {processed_count} processed, {skipped_count} skipped")
        return documents


def main():
    parser = argparse.ArgumentParser(description='Load filesystem documents into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, 
                       help='Path to directory to index recursively')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", 
                       help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Filesystem", 
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