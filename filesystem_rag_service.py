#!/usr/bin/env python3
"""
Filesystem RAG CLI Service

A command-line service for real-time filesystem monitoring, indexing, and searching
using Weaviate vector database. Combines the proven patterns from the web service
with comprehensive filesystem monitoring capabilities.

Usage:
    python filesystem_rag_service.py index /path/to/docs --class-prefix MyDocs
    python filesystem_rag_service.py search "query text" --class-prefix MyDocs
    python filesystem_rag_service.py monitor /path/to/docs --class-prefix MyDocs
    python filesystem_rag_service.py scan /path/to/docs --class-prefix MyDocs --schedule daily
"""

import asyncio
import argparse
import logging
import os
import re
import signal
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# LlamaIndex imports (following web service patterns)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, PromptHelper, GPTVectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

import weaviate

# Filesystem components
from indexing.weaviate_indexer import Indexer
from scraping.filesystem.filesystem_monitor import FilesystemMonitor
from indexing.filesystem.index_filesystem import FilesystemLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/filesystem_rag_{datetime.now().strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Constants (following web service patterns)
EMBED_MODEL_NAME = "text-embedding-3-large"
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1
SOURCE = "Filesystem"

# Default configuration
DEFAULT_WEAVIATE_URL = "http://localhost:8777"
DEFAULT_CLASS_PREFIX = "Filesystem"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SEARCH_ALPHA = 0.5
DEFAULT_NUM_RESULTS = 10


class FilesystemRAGService:
    """
    CLI service for filesystem-based RAG operations.
    
    Combines real-time filesystem monitoring with Weaviate indexing and search,
    following the proven patterns from the web service architecture.
    """
    
    def __init__(self, weaviate_url: str = DEFAULT_WEAVIATE_URL, debug: bool = False):
        self.weaviate_url = weaviate_url
        self.debug = debug
        self.weaviate_client = None
        self.monitor = None
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"Initialized Filesystem RAG Service with Weaviate at {weaviate_url}")
    
    def _get_weaviate_client(self):
        """Get Weaviate client (following web service pattern)."""
        if self.weaviate_client is None:
            self.weaviate_client = weaviate.Client(self.weaviate_url)
            
            if not self.weaviate_client.is_live():
                raise Exception(f"Weaviate is not live at {self.weaviate_url}")
            
            logger.info(f"Connected to Weaviate at {self.weaviate_url} (Version {self.weaviate_client.get_meta()['version']})")
        
        return self.weaviate_client
    
    def _get_query_engine(self, class_prefix: str, temperature: float = DEFAULT_TEMPERATURE,
                         search_alpha: float = DEFAULT_SEARCH_ALPHA, num_results: int = DEFAULT_NUM_RESULTS):
        """Get query engine for searching (following web service pattern)."""
        client = self._get_weaviate_client()
        
        logger.info(f"Creating query engine with parameters:")
        logger.info(f"  class_prefix: {class_prefix}")
        logger.info(f"  temperature: {temperature}")
        logger.info(f"  search_alpha: {search_alpha}")
        logger.info(f"  num_results: {num_results}")
        
        # Set up LLM and embedding model
        llm = OpenAI(model="gpt-4o", temperature=temperature)
        embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
        prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.prompt_helper = prompt_helper
        
        # Create vector store and index
        vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = GPTVectorStoreIndex([], storage_context=storage_context)
        
        # Configure retriever with hybrid search
        retriever = VectorIndexRetriever(
            index,
            similarity_top_k=num_results,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=search_alpha,
        )
        
        return RetrieverQueryEngine.from_args(retriever)
    
    def _escape_text(self, text: str) -> str:
        """Escape special characters for display (following web service pattern)."""
        text = re.sub("<", "&lt;", text)
        text = re.sub(">", "&gt;", text)
        text = re.sub("([_#])", r"\\\1", text)
        return text
    
    def _get_unique_nodes(self, nodes):
        """Get unique nodes from response (following web service pattern)."""
        docs_ids = set()
        unique_nodes = list()
        for node in nodes:
            if node.node.ref_doc_id not in docs_ids:
                docs_ids.add(node.node.ref_doc_id)
                unique_nodes.append(node)
        return unique_nodes
    
    def index_directory(self, directory_path: str, class_prefix: str = DEFAULT_CLASS_PREFIX, 
                       remove_existing: bool = False) -> None:
        """
        Index all supported files in a directory into Weaviate.
        
        Args:
            directory_path: Path to directory to index
            class_prefix: Weaviate class prefix
            remove_existing: Whether to remove existing data
        """
        logger.info(f"Starting directory indexing: {directory_path}")
        logger.info(f"Class prefix: {class_prefix}")
        
        try:
            # Load documents from filesystem
            loader = FilesystemLoader(directory_path, debug=self.debug)
            documents = loader.load_all_documents()
            
            if not documents:
                logger.warning("No documents were loaded. Check your input directory and supported file types.")
                return
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Index the documents in Weaviate
            indexer = Indexer(self.weaviate_url, class_prefix, remove_existing)
            indexer.index(documents)
            
            logger.info(f"Successfully indexed {len(documents)} documents into '{class_prefix}_Node'")
            
        except Exception as e:
            logger.error(f"Failed to index directory {directory_path}: {e}")
            raise
    
    def search_documents(self, query: str, class_prefix: str = DEFAULT_CLASS_PREFIX,
                        temperature: float = DEFAULT_TEMPERATURE, search_alpha: float = DEFAULT_SEARCH_ALPHA,
                        num_results: int = DEFAULT_NUM_RESULTS) -> str:
        """
        Search indexed documents using natural language query.
        
        Args:
            query: Search query
            class_prefix: Weaviate class prefix to search
            temperature: LLM temperature
            search_alpha: Hybrid search balance (0=keyword, 1=vector)
            num_results: Number of results to retrieve
            
        Returns:
            Formatted response with sources
        """
        logger.info(f"Searching for: '{query}'")
        logger.info(f"Class prefix: {class_prefix}")
        
        start_time = time.time()
        
        try:
            # Clean query
            query = re.sub('"', "", query)
            
            # Get query engine and execute search
            query_engine = self._get_query_engine(class_prefix, temperature, search_alpha, num_results)
            response = query_engine.query(query)
            
            end_time = time.time()
            logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
            
            # Format response with sources
            msg = f"{response.response}\n\nSources:\n\n"
            
            for node in self._get_unique_nodes(response.source_nodes):
                extra_info = node.node.extra_info
                text = node.node.text
                
                # Clean and truncate text
                text = re.sub(r"\n+", " ", text)
                text = textwrap.shorten(text, width=100, placeholder="...")
                text = self._escape_text(text)
                
                source = extra_info.get('source', 'Unknown')
                title = extra_info.get('title', 'Untitled')
                link = extra_info.get('link', '')
                
                if link:
                    msg += f"* {source}: [{title}]({link})\n  {text}\n\n"
                else:
                    msg += f"* {source}: {title}\n  {text}\n\n"
            
            return msg
            
        except Exception as e:
            error_msg = f"Search failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def monitor_directory(self, directory_path: str, class_prefix: str = DEFAULT_CLASS_PREFIX,
                              debounce: int = 1000) -> None:
        """
        Monitor a directory for real-time file changes and update index.
        
        Args:
            directory_path: Directory to monitor
            class_prefix: Weaviate class prefix
            debounce: Debounce interval in milliseconds
        """
        logger.info(f"Starting real-time monitoring of: {directory_path}")
        logger.info(f"Class prefix: {class_prefix}")
        
        def setup_signal_handlers():
            """Setup signal handlers for graceful shutdown."""
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, stopping monitor...")
                if self.monitor:
                    asyncio.create_task(self.monitor.stop())
                else:
                    sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        
        async def process_file_change(file_path: Path) -> None:
            """Process a file that was created or modified."""
            try:
                logger.info(f"Processing file change: {file_path}")
                
                # Import the file processor
                from scraping.filesystem.file_processor import process_file, is_supported_file_type
                
                # Check if file is supported
                if not is_supported_file_type(file_path):
                    logger.debug(f"Skipping unsupported file type: {file_path}")
                    return
                
                # Extract text content
                text_content = process_file(file_path)
                
                if not text_content or not text_content.strip():
                    logger.warning(f"No content extracted from: {file_path}")
                    return
                
                # Log first 20 chars being indexed for debugging
                preview = text_content[:20].replace('\n', '\\n').replace('\r', '\\r')
                logger.info(f"INDEXING {file_path.name}: First 20 chars = '{preview}'")
                
                # Create document
                loader = FilesystemLoader(file_path.parent, debug=self.debug)
                doc = loader.create_document(
                    file_path,
                    file_path.name,
                    f"file://{file_path.absolute()}",
                    text_content
                )
                
                # Index the document
                indexer = Indexer(self.weaviate_url, class_prefix, False)
                indexer.index([doc])
                
                logger.info(f"Successfully indexed updated file: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process file change {file_path}: {e}")
        
        async def handle_file_removal(file_path: Path) -> None:
            """Handle file removal (placeholder for future implementation)."""
            logger.info(f"File deleted: {file_path}")
            # TODO: Implement document removal from Weaviate
        
        setup_signal_handlers()
        
        try:
            directory_path_obj = Path(directory_path)
            self.monitor = FilesystemMonitor(
                directory_path_obj,
                debounce=debounce,
                process_file_func=process_file_change,
                remove_file_func=handle_file_removal
            )
            
            async with self.monitor:
                logger.info("Monitor started. Press Ctrl+C to stop.")
                await self.monitor.task
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            raise
        finally:
            logger.info("Monitor shutdown complete")
    
    def scan_directory_scheduled(self, directory_path: str, class_prefix: str = DEFAULT_CLASS_PREFIX,
                               schedule: str = "daily") -> None:
        """
        Perform scheduled scanning of directory (placeholder for future implementation).
        
        Args:
            directory_path: Directory to scan
            class_prefix: Weaviate class prefix  
            schedule: Schedule frequency (hourly, daily, weekly)
        """
        logger.info(f"Scheduled scanning not yet implemented")
        logger.info(f"Would scan {directory_path} on {schedule} schedule")
        logger.info(f"For now, performing one-time scan...")
        
        # For now, just perform a one-time index
        self.index_directory(directory_path, class_prefix, remove_existing=False)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Filesystem RAG CLI Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Examples:
              # Index a directory
              python filesystem_rag_service.py index /path/to/docs --class-prefix MyDocs
              
              # Search indexed documents
              python filesystem_rag_service.py search "How do I configure the system?" --class-prefix MyDocs
              
              # Monitor directory for real-time updates
              python filesystem_rag_service.py monitor /path/to/docs --class-prefix MyDocs
              
              # Scheduled scanning (future feature)
              python filesystem_rag_service.py scan /path/to/docs --class-prefix MyDocs --schedule daily
        ''')
    )
    
    parser.add_argument('-w', '--weaviate-url', type=str, default=DEFAULT_WEAVIATE_URL,
                       help=f'Weaviate database URL (default: {DEFAULT_WEAVIATE_URL})')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Enable debug logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index directory into Weaviate')
    index_parser.add_argument('directory', type=str, help='Directory to index')
    index_parser.add_argument('-c', '--class-prefix', type=str, default=DEFAULT_CLASS_PREFIX,
                             help=f'Weaviate class prefix (default: {DEFAULT_CLASS_PREFIX})')
    index_parser.add_argument('-r', '--remove-existing', action='store_true',
                             help='Remove existing data before indexing')
    
    # Search command  
    search_parser = subparsers.add_parser('search', help='Search indexed documents')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('-c', '--class-prefix', type=str, default=DEFAULT_CLASS_PREFIX,
                              help=f'Weaviate class prefix (default: {DEFAULT_CLASS_PREFIX})')
    search_parser.add_argument('-t', '--temperature', type=float, default=DEFAULT_TEMPERATURE,
                              help=f'LLM temperature (default: {DEFAULT_TEMPERATURE})')
    search_parser.add_argument('-a', '--search-alpha', type=float, default=DEFAULT_SEARCH_ALPHA,
                              help=f'Hybrid search balance 0-1 (default: {DEFAULT_SEARCH_ALPHA})')
    search_parser.add_argument('-n', '--num-results', type=int, default=DEFAULT_NUM_RESULTS,
                              help=f'Number of results (default: {DEFAULT_NUM_RESULTS})')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor directory for real-time updates')
    monitor_parser.add_argument('directory', type=str, help='Directory to monitor')
    monitor_parser.add_argument('-c', '--class-prefix', type=str, default=DEFAULT_CLASS_PREFIX,
                               help=f'Weaviate class prefix (default: {DEFAULT_CLASS_PREFIX})')
    monitor_parser.add_argument('--debounce', type=int, default=1000,
                               help='Debounce interval in milliseconds (default: 1000)')
    
    # Scan command (future feature)
    scan_parser = subparsers.add_parser('scan', help='Scheduled directory scanning (future feature)')
    scan_parser.add_argument('directory', type=str, help='Directory to scan')
    scan_parser.add_argument('-c', '--class-prefix', type=str, default=DEFAULT_CLASS_PREFIX,
                            help=f'Weaviate class prefix (default: {DEFAULT_CLASS_PREFIX})')
    scan_parser.add_argument('-s', '--schedule', type=str, choices=['hourly', 'daily', 'weekly'],
                            default='daily', help='Schedule frequency (default: daily)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize service
    service = FilesystemRAGService(args.weaviate_url, args.debug)
    
    try:
        if args.command == 'index':
            service.index_directory(args.directory, args.class_prefix, args.remove_existing)
            print(f"✓ Successfully indexed directory: {args.directory}")
            
        elif args.command == 'search':
            result = service.search_documents(
                args.query, args.class_prefix, args.temperature, 
                args.search_alpha, args.num_results
            )
            print(f"\n{result}")
            
        elif args.command == 'monitor':
            print(f"Starting real-time monitoring of {args.directory}")
            print("Press Ctrl+C to stop...")
            asyncio.run(service.monitor_directory(args.directory, args.class_prefix, args.debounce))
            
        elif args.command == 'scan':
            service.scan_directory_scheduled(args.directory, args.class_prefix, args.schedule)
            print(f"✓ Scheduled scan completed for: {args.directory}")
            
    except KeyboardInterrupt:
        print("\n✓ Operation interrupted by user")
    except Exception as e:
        print(f"✗ Error: {e}")
        if args.debug:
            logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()