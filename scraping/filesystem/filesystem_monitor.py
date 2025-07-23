"""
Filesystem monitoring module using watchfiles for high-performance file watching.

This module provides asynchronous filesystem monitoring capabilities optimized for
local RAG applications that need real-time updates when files are added, modified, or deleted.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Set, Callable, Optional, Any
from datetime import datetime

from watchfiles import awatch, Change
from .file_processor import (
    process_file as process_file_impl,
    get_supported_extensions,
    is_supported_file_type,
    get_processing_method
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/filesystem_monitor_{datetime.now().strftime("%Y%m%d")}.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Get supported extensions from the file processor
SUPPORTED_EXTENSIONS: Set[str] = get_supported_extensions()


def is_hidden(path: Path) -> bool:
    """
    Check if a file or any of its parent directories are hidden (start with '.').
    
    Args:
        path: Path to check
        
    Returns:
        True if the path or any parent is hidden
    """
    return any(part.startswith('.') for part in path.parts)


def is_supported_extension(path: Path) -> bool:
    """
    Check if the file has a supported extension.
    
    Args:
        path: Path to check
        
    Returns:
        True if the file extension is supported
    """
    return is_supported_file_type(path)


def should_process_file(path: Path) -> bool:
    """
    Determine if a file should be processed based on extension and visibility.
    
    Args:
        path: Path to check
        
    Returns:
        True if the file should be processed
    """
    return (
        path.is_file() 
        and is_supported_extension(path) 
        and not is_hidden(path)
    )


async def process_file(path: Path) -> str:
    """
    Process a file that has been created or modified.
    Extracts text content from the file using appropriate processing methods.
    
    Args:
        path: Path to the file that was created or modified
        
    Returns:
        Extracted text content as string
        
    Raises:
        Exception: If file processing fails
    """
    try:
        # Get processing method for logging
        method = get_processing_method(path)
        logger.info(f"Processing file: {path} (method: {method})")
        
        # Process the file and extract text
        text_content = process_file_impl(path)
        
        # Log extraction statistics
        char_count = len(text_content)
        word_count = len(text_content.split())
        logger.info(f"Extracted {char_count} characters, {word_count} words from {path.name}")
        
        # TODO: Here you can add additional processing such as:
        # - Generating embeddings
        # - Storing in vector database
        # - Updating search indices
        
        return text_content
        
    except Exception as e:
        logger.error(f"Failed to process file {path}: {e}")
        raise


async def remove_file(path: Path) -> None:
    """
    Handle removal of a file from the monitoring system.
    This is a placeholder function that should be replaced with actual removal logic.
    
    Args:
        path: Path to the file that was deleted
    """
    logger.info(f"Removing file: {path}")
    # TODO: Implement actual file removal logic here
    # This could include:
    # - Removing from vector database
    # - Cleaning up associated metadata
    # - Updating indices
    pass


async def watch_directory(
    root: Path, 
    stop_event: Optional[asyncio.Event] = None,
    debounce: int = 1000,
    process_file_func: Optional[Callable[[Path], Any]] = None,
    remove_file_func: Optional[Callable[[Path], Any]] = None
) -> None:
    """
    Watch a directory recursively for file changes using watchfiles.
    
    Args:
        root: Root directory to watch
        stop_event: Optional event to stop watching
        debounce: Milliseconds to debounce events (default: 1000ms)
        process_file_func: Optional custom function to process files
        remove_file_func: Optional custom function to handle file removal
    """
    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")
    
    if not root.is_dir():
        raise ValueError(f"Root path is not a directory: {root}")
    
    # Use custom functions if provided, otherwise use defaults
    process_func = process_file_func or process_file
    remove_func = remove_file_func or remove_file
    
    logger.info(f"Starting filesystem watcher for: {root.absolute()}")
    logger.info(f"Monitoring extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    logger.info(f"Debounce interval: {debounce}ms")
    
    processed_files = 0
    removed_files = 0
    
    try:
        async for changes in awatch(
            str(root), 
            debounce=debounce, 
            recursive=True,
            stop_event=stop_event
        ):
            for change_type, file_path in changes:
                path = Path(file_path)
                
                # Log all detected changes for debugging
                logger.debug(f"Detected {change_type.name}: {path}")
                
                # Skip if file should not be processed
                if not should_process_file(path):
                    logger.debug(f"Skipping {path}: {'hidden' if is_hidden(path) else 'unsupported extension'}")
                    continue
                
                try:
                    if change_type in (Change.added, Change.modified):
                        logger.info(f"File {change_type.name}: {path}")
                        if asyncio.iscoroutinefunction(process_func):
                            await process_func(path)
                        else:
                            process_func(path)
                        processed_files += 1
                        
                    elif change_type == Change.deleted:
                        logger.info(f"File deleted: {path}")
                        if asyncio.iscoroutinefunction(remove_func):
                            await remove_func(path)
                        else:
                            remove_func(path)
                        removed_files += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}", exc_info=True)
                    
    except asyncio.CancelledError:
        logger.info("Filesystem watcher cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in filesystem watcher: {e}", exc_info=True)
        raise
    finally:
        logger.info(f"Filesystem watcher stopped. Processed: {processed_files}, Removed: {removed_files}")


class FilesystemMonitor:
    """
    A class-based interface for filesystem monitoring with lifecycle management.
    """
    
    def __init__(
        self, 
        root: Path, 
        debounce: int = 1000,
        process_file_func: Optional[Callable[[Path], Any]] = None,
        remove_file_func: Optional[Callable[[Path], Any]] = None
    ):
        self.root = Path(root)
        self.debounce = debounce
        self.process_file_func = process_file_func
        self.remove_file_func = remove_file_func
        self.stop_event = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def start(self) -> None:
        """Start the filesystem monitor."""
        if self.task is not None:
            raise RuntimeError("Monitor is already running")
            
        self.stop_event.clear()
        self.task = asyncio.create_task(
            watch_directory(
                self.root, 
                self.stop_event, 
                self.debounce,
                self.process_file_func,
                self.remove_file_func
            )
        )
        self.logger.info(f"Started filesystem monitor for {self.root}")
        
    async def stop(self) -> None:
        """Stop the filesystem monitor."""
        if self.task is None:
            return
            
        self.stop_event.set()
        
        try:
            await asyncio.wait_for(self.task, timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Monitor did not stop gracefully, cancelling task")
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        self.task = None
        self.logger.info("Stopped filesystem monitor")
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Global variable to store the monitor for signal handling
_current_monitor: Optional[FilesystemMonitor] = None


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown in Docker containers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if _current_monitor:
            # Create a new event loop if we're not in one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Schedule the stop coroutine
            asyncio.create_task(_current_monitor.stop())
        else:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """
    Example usage of the filesystem monitor.
    This can be used as a standalone script or as a reference implementation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor filesystem for changes')
    parser.add_argument('directory', type=Path, help='Directory to monitor')
    parser.add_argument('--debounce', type=int, default=1000, 
                       help='Debounce interval in milliseconds (default: 1000)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    global _current_monitor
    
    try:
        async with FilesystemMonitor(
            args.directory, 
            debounce=args.debounce
        ) as monitor:
            _current_monitor = monitor
            logger.info("Filesystem monitor started. Press Ctrl+C to stop.")
            
            # Keep the monitor running
            await monitor.task
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        _current_monitor = None
        logger.info("Filesystem monitor shutdown complete")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    asyncio.run(main())