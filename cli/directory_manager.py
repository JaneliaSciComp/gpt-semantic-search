"""
Directory manager for persistent monitoring and indexing.

Handles concurrent monitoring of multiple directories with automatic
indexing and real-time file change processing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from .config import RAGConfig, DirectoryConfig
from scraping.filesystem.filesystem_monitor import FilesystemMonitor
from scraping.filesystem.file_processor import process_file, is_supported_file_type
from indexing.weaviate_indexer import Indexer
from indexing.filesystem.index_filesystem import FilesystemLoader

logger = logging.getLogger(__name__)


class DirectoryMonitor:
    """Individual directory monitor with indexing capabilities."""
    
    def __init__(
        self,
        directory_config: DirectoryConfig,
        weaviate_url: str,
        debounce_ms: int = 1000,
        debug: bool = False
    ):
        self.config = directory_config
        self.weaviate_url = weaviate_url
        self.debounce_ms = debounce_ms
        self.debug = debug
        
        self.monitor: Optional[FilesystemMonitor] = None
        self.is_running = False
        
        # Statistics
        self.files_processed = 0
        self.files_failed = 0
        self.last_activity = None
    
    async def start(self) -> None:
        """Start monitoring the directory."""
        if self.is_running:
            logger.warning(f"Monitor already running for: {self.config.path}")
            return
        
        if not self.config.enabled:
            logger.info(f"Skipping disabled directory: {self.config.path}")
            return
        
        try:
            logger.info(f"Starting monitor for: {self.config.path}")
            
            directory_path = Path(self.config.path)
            self.monitor = FilesystemMonitor(
                directory_path,
                debounce=self.debounce_ms,
                process_file_func=self._process_file_change,
                remove_file_func=self._handle_file_removal
            )
            
            self.is_running = True
            logger.info(f"✓ Monitor started: {self.config.path} ({self.config.class_prefix})")
            
        except Exception as e:
            logger.error(f"Failed to start monitor for {self.config.path}: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop monitoring the directory."""
        if not self.is_running:
            return
        
        try:
            logger.info(f"Stopping monitor for: {self.config.path}")
            
            if self.monitor:
                await self.monitor.stop()
                self.monitor = None
            
            self.is_running = False
            logger.info(f"✓ Monitor stopped: {self.config.path}")
            
        except Exception as e:
            logger.error(f"Failed to stop monitor for {self.config.path}: {e}")
    
    async def _process_file_change(self, file_path: Path) -> None:
        """Process a file that was created or modified."""
        try:
            logger.info(f"Processing file change: {file_path}")
            self.last_activity = datetime.now()
            
            # Check if file is supported
            if not is_supported_file_type(file_path):
                logger.debug(f"Skipping unsupported file type: {file_path}")
                return
            
            # Extract text content
            text_content = process_file(file_path)
            
            if not text_content or not text_content.strip():
                logger.warning(f"No content extracted from: {file_path}")
                self.files_failed += 1
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
            indexer = Indexer(self.weaviate_url, self.config.class_prefix, False)
            indexer.index([doc])
            
            self.files_processed += 1
            logger.info(f"Successfully indexed: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process file change {file_path}: {e}")
            self.files_failed += 1
    
    async def _handle_file_removal(self, file_path: Path) -> None:
        """Handle file removal (placeholder for future implementation)."""
        logger.info(f"File deleted: {file_path}")
        self.last_activity = datetime.now()
        # TODO: Implement document removal from Weaviate
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        return {
            "path": self.config.path,
            "class_prefix": self.config.class_prefix,
            "enabled": self.config.enabled,
            "running": self.is_running,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }
    
    def __str__(self) -> str:
        status = "✓ Running" if self.is_running else ("✗ Disabled" if not self.config.enabled else "✗ Stopped")
        return f"{self.config.path} ({self.config.class_prefix}) - {status}"


class DirectoryManager:
    """
    Manages multiple directory monitors with persistent configuration.
    
    Handles concurrent monitoring of multiple directories, automatic indexing,
    and configuration persistence across sessions.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.monitors: Dict[str, DirectoryMonitor] = {}
        self.is_running = False
        
        # Event callbacks
        self.on_file_processed: Optional[Callable[[str, Path], None]] = None
        self.on_monitor_error: Optional[Callable[[str, Exception], None]] = None
    
    async def start_all(self) -> None:
        """Start monitoring all configured directories."""
        if self.is_running:
            logger.warning("Directory manager already running")
            return
        
        logger.info("Starting directory manager...")
        
        # Create monitors for all enabled directories
        for dir_config in self.config.list_directories(enabled_only=True):
            await self.add_monitor(dir_config)
        
        self.is_running = True
        logger.info(f"✓ Directory manager started with {len(self.monitors)} monitors")
    
    async def stop_all(self) -> None:
        """Stop all directory monitors."""
        if not self.is_running:
            return
        
        logger.info("Stopping directory manager...")
        
        # Stop all monitors
        for path in list(self.monitors.keys()):
            await self.remove_monitor(path)
        
        self.is_running = False
        logger.info("✓ Directory manager stopped")
    
    async def add_monitor(self, dir_config: DirectoryConfig) -> DirectoryMonitor:
        """
        Add a new directory monitor.
        
        Args:
            dir_config: Directory configuration
            
        Returns:
            Created DirectoryMonitor instance
        """
        if dir_config.path in self.monitors:
            logger.warning(f"Monitor already exists for: {dir_config.path}")
            return self.monitors[dir_config.path]
        
        try:
            monitor = DirectoryMonitor(
                dir_config,
                self.config.weaviate_url,
                debounce_ms=self.config.get_setting("debounce_ms", 1000),
                debug=self.config.get_setting("debug", False)
            )
            
            await monitor.start()
            self.monitors[dir_config.path] = monitor
            
            logger.info(f"Added monitor: {monitor}")
            return monitor
            
        except Exception as e:
            logger.error(f"Failed to add monitor for {dir_config.path}: {e}")
            if self.on_monitor_error:
                self.on_monitor_error(dir_config.path, e)
            raise
    
    async def remove_monitor(self, path: str) -> bool:
        """
        Remove a directory monitor.
        
        Args:
            path: Directory path
            
        Returns:
            True if removed, False if not found
        """
        abs_path = str(Path(path).absolute())
        
        if abs_path not in self.monitors:
            logger.warning(f"No monitor found for: {abs_path}")
            return False
        
        try:
            monitor = self.monitors[abs_path]
            await monitor.stop()
            
            del self.monitors[abs_path]
            logger.info(f"Removed monitor: {abs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove monitor for {abs_path}: {e}")
            return False
    
    async def restart_monitor(self, path: str) -> bool:
        """
        Restart a specific directory monitor.
        
        Args:
            path: Directory path
            
        Returns:
            True if restarted successfully
        """
        abs_path = str(Path(path).absolute())
        
        if abs_path not in self.monitors:
            logger.warning(f"No monitor found for: {abs_path}")
            return False
        
        try:
            monitor = self.monitors[abs_path]
            await monitor.stop()
            await monitor.start()
            
            logger.info(f"Restarted monitor: {abs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart monitor for {abs_path}: {e}")
            return False
    
    def get_monitor(self, path: str) -> Optional[DirectoryMonitor]:
        """Get monitor by directory path."""
        abs_path = str(Path(path).absolute())
        return self.monitors.get(abs_path)
    
    def list_monitors(self) -> List[DirectoryMonitor]:
        """Get list of all monitors."""
        return list(self.monitors.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status of all monitors."""
        monitor_statuses = []
        total_processed = 0
        total_failed = 0
        running_count = 0
        
        for monitor in self.monitors.values():
            status = monitor.get_status()
            monitor_statuses.append(status)
            total_processed += status["files_processed"]
            total_failed += status["files_failed"]
            if status["running"]:
                running_count += 1
        
        return {
            "total_monitors": len(self.monitors),
            "running_monitors": running_count,
            "total_files_processed": total_processed,
            "total_files_failed": total_failed,
            "monitors": monitor_statuses
        }
    
    async def index_directory(
        self,
        path: str,
        remove_existing: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Perform initial indexing of a directory.
        
        Args:
            path: Directory path to index
            remove_existing: Whether to remove existing index
            progress_callback: Optional callback for progress updates
            
        Returns:
            Number of documents indexed
        """
        abs_path = str(Path(path).absolute())
        dir_config = self.config.get_directory(abs_path)
        
        if not dir_config:
            raise ValueError(f"Directory not configured: {abs_path}")
        
        logger.info(f"Starting initial index of: {abs_path}")
        
        try:
            # Load documents from filesystem
            loader = FilesystemLoader(
                abs_path,
                debug=self.config.get_setting("debug", False)
            )
            documents = loader.load_all_documents()
            
            if not documents:
                logger.warning(f"No documents found in: {abs_path}")
                return 0
            
            # Index the documents
            indexer = Indexer(
                self.config.weaviate_url,
                dir_config.class_prefix,
                remove_existing
            )
            
            if progress_callback:
                progress_callback(0, len(documents))
            
            indexer.index(documents)
            
            if progress_callback:
                progress_callback(len(documents), len(documents))
            
            # Update last indexed time
            self.config.update_last_indexed(abs_path)
            
            logger.info(f"✓ Indexed {len(documents)} documents from: {abs_path}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to index directory {abs_path}: {e}")
            raise
    
    def __str__(self) -> str:
        status = "✓ Running" if self.is_running else "✗ Stopped"
        return f"DirectoryManager - {status} ({len(self.monitors)} monitors)"