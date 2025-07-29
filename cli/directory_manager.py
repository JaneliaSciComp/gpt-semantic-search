"""
Directory manager for persistent monitoring and indexing.

Handles concurrent monitoring of multiple directories with automatic
indexing and real-time file change processing.
"""

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from .config import RAGConfig, DirectoryConfig
from scraping.filesystem.filesystem_monitor import FilesystemMonitor
from scraping.filesystem.file_processor import process_file, is_supported_file_type
from indexing.weaviate_indexer import Indexer
from indexing.filesystem.index_filesystem import FilesystemLoader, StreamingFilesystemProcessor

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
            return
        
        if not self.config.enabled:
            logger.info(f"Directory {self.config.path} is disabled, skipping monitor start")
            return
        
        try:
            directory_path = Path(self.config.path)
            
            # Verify directory exists
            if not directory_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            if not directory_path.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            logger.info(f"Starting filesystem monitor for: {directory_path}")
            
            # Create the monitor
            self.monitor = FilesystemMonitor(
                directory_path,
                debounce=self.debounce_ms,
                process_file_func=self._process_file_change,
                remove_file_func=self._handle_file_removal
            )
            
            # CRITICAL FIX: Actually start the monitor task
            await self.monitor.start()
            
            # Verify the monitor task is running
            if self.monitor.task is None or self.monitor.task.done():
                raise RuntimeError("Monitor task failed to start properly")
            
            self.is_running = True
            logger.info(f"✓ Filesystem monitor started successfully for {directory_path}")
            
        except Exception as e:
            logger.error(f"Failed to start monitor for {self.config.path}: {e}")
            self.is_running = False
            # Clean up if monitor was partially created
            if self.monitor:
                try:
                    await self.monitor.stop()
                except:
                    pass
                self.monitor = None
            raise
    
    async def stop(self) -> None:
        """Stop monitoring the directory."""
        if not self.is_running:
            return
        
        try:
            logger.info(f"Stopping filesystem monitor for: {self.config.path}")
            
            if self.monitor:
                await self.monitor.stop()
                self.monitor = None
            
            self.is_running = False
            logger.info(f"✓ Filesystem monitor stopped for {self.config.path}")
            
        except Exception as e:
            logger.error(f"Failed to stop monitor for {self.config.path}: {e}")
            # Force cleanup even if stop failed
            self.is_running = False
            self.monitor = None
    
    async def _process_file_change(self, file_path: Path) -> None:
        """Process a file that was created or modified with intelligent change detection."""
        try:
            self.last_activity = datetime.now()
            
            # Check if file is supported
            if not is_supported_file_type(file_path):
                logger.debug(f"Skipping unsupported file type: {file_path}")
                return
            
            # Get current file metadata for change detection
            try:
                file_stat = file_path.stat()
                current_mtime = file_stat.st_mtime
                current_size = file_stat.st_size
            except OSError as e:
                logger.warning(f"Cannot access file stats for {file_path}: {e}")
                return
            
            # Generate current file hash for content change detection
            try:
                file_hash = hashlib.md5()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        file_hash.update(chunk)
                current_hash = file_hash.hexdigest()
            except OSError as e:
                logger.warning(f"Cannot read file for hashing {file_path}: {e}")
                return
            
            # Check if document already exists in Weaviate
            indexer = Indexer(self.weaviate_url, "Janelia", False)
            existing_docs = indexer.get_all_filesystem_documents()
            
            # Find existing document for this file path
            existing_doc = None
            for doc in existing_docs:
                if doc.get("file_path") == str(file_path):
                    existing_doc = doc
                    break
            
            # Determine if we need to re-index
            should_reindex = True
            if existing_doc:
                existing_hash = existing_doc.get("file_hash")
                existing_mtime = existing_doc.get("file_mtime")
                
                if (existing_hash == current_hash and 
                    existing_mtime == current_mtime):
                    logger.debug(f"File unchanged, skipping re-indexing: {file_path.name}")
                    should_reindex = False
                else:
                    logger.info(f"File changed, re-indexing: {file_path.name}")
                    # Delete old version first
                    indexer.delete_document_by_path(str(file_path))
            else:
                logger.info(f"New file detected, indexing: {file_path.name}")
            
            if should_reindex:
                # Extract text content
                text_content = process_file(file_path)
                
                if not text_content or not text_content.strip():
                    logger.warning(f"No content extracted from: {file_path}")
                    self.files_failed += 1
                    return
                
                # Import required modules
                from llama_index.core import Document
                from scraping.filesystem.file_processor import get_processing_method
                import os
                import hashlib
                
                # Create document with enhanced metadata
                metadata = {
                    "title": file_path.name,
                    "link": f"file://{file_path.absolute()}",
                    "source": "Filesystem",
                    "file_path": str(file_path),
                    "processing_method": get_processing_method(file_path),
                    "file_mtime": current_mtime,
                    "file_size": current_size,
                    "file_hash": current_hash,
                    "indexed_at": datetime.now().isoformat()
                }
                doc = Document(text=text_content, doc_id=str(file_path), metadata=metadata)
                
                # Index the document
                indexer.index([doc])
                logger.info(f"Successfully indexed {file_path.name}")
            
            self.files_processed += 1
            
        except Exception as e:
            logger.error(f"Failed to process file change {file_path}: {e}")
            self.files_failed += 1
    
    async def _handle_file_removal(self, file_path: Path) -> None:
        """Handle file removal - delete document from Weaviate."""
        try:
            self.last_activity = datetime.now()
            
            # Check if it was a supported file type
            if not is_supported_file_type(file_path):
                logger.debug(f"Skipping removal of unsupported file type: {file_path}")
                return
            
            # Delete the document from Weaviate
            indexer = Indexer(self.weaviate_url, "Janelia", False)
            success = indexer.delete_document_by_path(str(file_path))
            
            if success:
                logger.info(f"Successfully removed document for deleted file: {file_path.name}")
            else:
                logger.debug(f"No document found to remove for: {file_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to handle file removal {file_path}: {e}")
            self.files_failed += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        monitor_healthy = False
        monitor_task_status = "not_created"
        
        if self.monitor:
            if self.monitor.task:
                if self.monitor.task.done():
                    monitor_task_status = "completed"
                    if self.monitor.task.exception():
                        monitor_task_status = f"failed: {self.monitor.task.exception()}"
                elif self.monitor.task.cancelled():
                    monitor_task_status = "cancelled"
                else:
                    monitor_task_status = "running"
                    monitor_healthy = True
            else:
                monitor_task_status = "no_task"
        
        return {
            "path": self.config.path,
            "class_prefix": self.config.class_prefix,
            "enabled": self.config.enabled,
            "running": self.is_running,
            "monitor_healthy": monitor_healthy,
            "monitor_task_status": monitor_task_status,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }
    
    def is_monitor_healthy(self) -> bool:
        """Check if the monitor is healthy and running."""
        if not self.is_running or not self.monitor:
            return False
        
        if not self.monitor.task:
            return False
        
        # Task should be running (not done, not cancelled)
        return not self.monitor.task.done() and not self.monitor.task.cancelled()
    
    async def restart_if_unhealthy(self) -> bool:
        """Restart the monitor if it's unhealthy."""
        if self.is_monitor_healthy():
            return False  # Already healthy
        
        logger.warning(f"Monitor for {self.config.path} is unhealthy, attempting restart...")
        
        try:
            # Stop current monitor
            await self.stop()
            
            # Start fresh monitor  
            await self.start()
            
            logger.info(f"✓ Successfully restarted monitor for {self.config.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart monitor for {self.config.path}: {e}")
            return False
    
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
            return
        
        # Create monitors for all enabled directories
        for dir_config in self.config.list_directories(enabled_only=True):
            await self.add_monitor(dir_config)
        
        self.is_running = True
    
    async def stop_all(self) -> None:
        """Stop all directory monitors."""
        if not self.is_running:
            return
        
        # Stop all monitors
        for path in list(self.monitors.keys()):
            await self.remove_monitor(path)
        
        self.is_running = False
    
    async def add_monitor(self, dir_config: DirectoryConfig) -> DirectoryMonitor:
        """
        Add a new directory monitor.
        
        Args:
            dir_config: Directory configuration
            
        Returns:
            Created DirectoryMonitor instance
        """
        if dir_config.path in self.monitors:
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
            return False
        
        try:
            monitor = self.monitors[abs_path]
            await monitor.stop()
            
            del self.monitors[abs_path]
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
            return False
        
        try:
            monitor = self.monitors[abs_path]
            await monitor.stop()
            await monitor.start()
            
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
    
    async def index_directory_streaming(
        self,
        path: str,
        remove_existing: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> int:
        """
        Perform streaming indexing of a directory with progressive updates.
        
        Args:
            path: Directory path to index
            remove_existing: Whether to remove existing index  
            progress_callback: Optional callback for progress updates (stage, current, total)
            
        Returns:
            Number of documents indexed
        """
        abs_path = str(Path(path).absolute())
        dir_config = self.config.get_directory(abs_path)
        
        if not dir_config:
            raise ValueError(f"Directory not configured: {abs_path}")
        
        try:
            # Create streaming processor
            processor = StreamingFilesystemProcessor(
                abs_path,
                batch_size=20,  # Optimize for API rate limits
                max_workers=5,  # Concurrent file processing
                debug=self.config.get_setting("debug", False)
            )
            
            # Create indexer for progressive batch indexing using unified class prefix
            indexer = Indexer(
                self.config.weaviate_url,
                "Janelia",
                remove_existing
            )
            
            total_documents = 0
            batch_number = 0
            
            # Process and index documents in batches
            async for document_batch in processor.stream_document_batches(progress_callback):
                batch_number += 1
                
                # Index this batch immediately
                logger.info(f"INDEXING: Starting batch {batch_number} with {len(document_batch)} documents")
                
                try:
                    indexer.index_batch(
                        document_batch,
                        batch_number=batch_number,
                        total_batches=batch_number,  # We don't know total ahead of time
                        progress_callback=progress_callback
                    )
                    
                    total_documents += len(document_batch)
                    
                    # Log progress
                    logger.info(f"INDEXING: Batch {batch_number} complete - {total_documents} total documents indexed and searchable")
                    
                except Exception as e:
                    logger.error(f"INDEXING: Failed to index batch {batch_number}: {e}")
                    # Continue with next batch rather than failing completely
                    continue
            
            # Update last indexed time
            self.config.update_last_indexed(abs_path)
            
            logger.info(f"INDEXING: Complete - {total_documents} documents indexed from {abs_path}")
            return total_documents
            
        except Exception as e:
            logger.error(f"Failed to index directory {abs_path}: {e}")
            raise

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
            
            # Index the documents using unified class prefix
            indexer = Indexer(
                self.config.weaviate_url,
                "Janelia",
                remove_existing
            )
            
            if progress_callback:
                progress_callback(0, len(documents))
            
            indexer.index(documents)
            
            if progress_callback:
                progress_callback(len(documents), len(documents))
            
            # Update last indexed time
            self.config.update_last_indexed(abs_path)
            
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to index directory {abs_path}: {e}")
            raise
    
    async def check_monitor_health(self) -> Dict[str, Any]:
        """Check health of all monitors and report status."""
        healthy_monitors = []
        unhealthy_monitors = []
        
        for path, monitor in self.monitors.items():
            if monitor.is_monitor_healthy():
                healthy_monitors.append(path)
            else:
                unhealthy_monitors.append({
                    "path": path,
                    "status": monitor.get_status()
                })
        
        return {
            "total_monitors": len(self.monitors),
            "healthy_monitors": len(healthy_monitors),
            "unhealthy_monitors": len(unhealthy_monitors),
            "healthy_paths": healthy_monitors,
            "unhealthy_details": unhealthy_monitors
        }
    
    async def restart_unhealthy_monitors(self) -> Dict[str, Any]:
        """Restart all unhealthy monitors."""
        health_report = await self.check_monitor_health()
        
        if health_report["unhealthy_monitors"] == 0:
            logger.info("✅ All monitors are healthy, no restarts needed")
            return {"restarted": 0, "failed": 0, "details": []}
        
        logger.info(f"🔧 Attempting to restart {health_report['unhealthy_monitors']} unhealthy monitors...")
        
        restarted = 0
        failed = 0
        details = []
        
        for path, monitor in self.monitors.items():
            if not monitor.is_monitor_healthy():
                try:
                    success = await monitor.restart_if_unhealthy()
                    if success:
                        restarted += 1
                        details.append({"path": path, "status": "restarted"})
                    else:
                        failed += 1
                        details.append({"path": path, "status": "restart_failed"})
                except Exception as e:
                    failed += 1
                    details.append({"path": path, "status": f"restart_error: {e}"})
        
        logger.info(f"📊 Monitor restart complete: {restarted} restarted, {failed} failed")
        
        return {
            "restarted": restarted,
            "failed": failed,
            "details": details
        }
    
    async def sync_filesystem_with_database(self) -> Dict[str, Any]:
        """
        Periodic sync to compare filesystem state with Weaviate database.
        
        Returns:
            Sync results with counts of added, updated, and removed documents
        """
        logger.info("🔄 Starting filesystem-database sync...")
        
        sync_results = {
            "files_added": 0,
            "files_updated": 0,
            "files_removed": 0,
            "files_unchanged": 0,
            "errors": []
        }
        
        try:
            # Get all filesystem documents from Weaviate
            indexer = Indexer(self.config.weaviate_url, "Janelia", False)
            weaviate_docs = indexer.get_all_filesystem_documents()
            
            # Create lookup of Weaviate documents by file path
            weaviate_files = {doc["file_path"]: doc for doc in weaviate_docs}
            
            # Get all current filesystem files from monitored directories
            current_files = set()
            for dir_config in self.config.list_directories(enabled_only=True):
                dir_path = Path(dir_config.path)
                if dir_path.exists():
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file() and is_supported_file_type(file_path):
                            current_files.add(str(file_path))
            
            # Find files that exist in filesystem but not in Weaviate (NEW FILES)
            files_to_add = current_files - set(weaviate_files.keys())
            logger.info(f"Found {len(files_to_add)} new files to add")
            
            # Find files that exist in Weaviate but not in filesystem (DELETED FILES)
            files_to_remove = set(weaviate_files.keys()) - current_files
            logger.info(f"Found {len(files_to_remove)} deleted files to remove")
            
            # Find files that exist in both and check for changes (UPDATED FILES)
            files_to_check = current_files & set(weaviate_files.keys())
            logger.info(f"Found {len(files_to_check)} existing files to check for changes")
            
            # Process new files
            for file_path_str in files_to_add:
                try:
                    file_path = Path(file_path_str)
                    await self._process_single_file_for_sync(file_path, indexer)
                    sync_results["files_added"] += 1
                except Exception as e:
                    logger.error(f"Error adding file {file_path_str}: {e}")
                    sync_results["errors"].append(f"Add {file_path_str}: {e}")
            
            # Remove deleted files
            if files_to_remove:
                removed_count = indexer.delete_documents_by_paths(list(files_to_remove))
                sync_results["files_removed"] = removed_count
            
            # Check existing files for changes
            for file_path_str in files_to_check:
                try:
                    file_path = Path(file_path_str)
                    weaviate_doc = weaviate_files[file_path_str]
                    
                    # Check if file has changed
                    if await self._has_file_changed(file_path, weaviate_doc):
                        # Delete old version and re-index
                        indexer.delete_document_by_path(file_path_str)
                        await self._process_single_file_for_sync(file_path, indexer)
                        sync_results["files_updated"] += 1
                    else:
                        sync_results["files_unchanged"] += 1
                        
                except Exception as e:
                    logger.error(f"Error checking file {file_path_str}: {e}")
                    sync_results["errors"].append(f"Check {file_path_str}: {e}")
            
            logger.info(f"✅ Filesystem-database sync complete: "
                       f"{sync_results['files_added']} added, "
                       f"{sync_results['files_updated']} updated, "
                       f"{sync_results['files_removed']} removed, "
                       f"{sync_results['files_unchanged']} unchanged")
            
        except Exception as e:
            logger.error(f"Error during filesystem-database sync: {e}")
            sync_results["errors"].append(f"Sync error: {e}")
        
        return sync_results
    
    async def _process_single_file_for_sync(self, file_path: Path, indexer) -> None:
        """Process a single file for sync (helper method)."""
        # Extract text content
        text_content = process_file(file_path)
        
        if not text_content or not text_content.strip():
            logger.warning(f"No content extracted from: {file_path}")
            return
        
        # Get file metadata
        file_stat = file_path.stat()
        current_mtime = file_stat.st_mtime
        current_size = file_stat.st_size
        
        # Generate file hash
        file_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        current_hash = file_hash.hexdigest()
        
        # Create document
        from llama_index.core import Document
        from scraping.filesystem.file_processor import get_processing_method
        
        metadata = {
            "title": file_path.name,
            "link": f"file://{file_path.absolute()}",
            "source": "Filesystem",
            "file_path": str(file_path),
            "processing_method": get_processing_method(file_path),
            "file_mtime": current_mtime,
            "file_size": current_size,
            "file_hash": current_hash,
            "indexed_at": datetime.now().isoformat()
        }
        doc = Document(text=text_content, doc_id=str(file_path), metadata=metadata)
        
        # Index the document
        indexer.index([doc])
    
    async def _has_file_changed(self, file_path: Path, weaviate_doc: Dict) -> bool:
        """Check if a file has changed compared to its Weaviate document."""
        try:
            # Get current file metadata
            file_stat = file_path.stat()
            current_mtime = file_stat.st_mtime
            current_size = file_stat.st_size
            
            # Compare with stored metadata
            stored_mtime = weaviate_doc.get("file_mtime")
            stored_size = weaviate_doc.get("file_size")
            
            # Quick check: if mtime and size are the same, file likely unchanged
            if stored_mtime == current_mtime and stored_size == current_size:
                return False
            
            # If mtime or size changed, verify with hash
            file_hash = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
            current_hash = file_hash.hexdigest()
            
            stored_hash = weaviate_doc.get("file_hash")
            return current_hash != stored_hash
            
        except OSError:
            # If we can't read the file, assume it changed
            return True
    
    def __str__(self) -> str:
        status = "✓ Running" if self.is_running else "✗ Stopped"
        return f"DirectoryManager - {status} ({len(self.monitors)} monitors)"