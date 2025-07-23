"""
Interactive session manager for filesystem RAG CLI.

Handles the main REPL loop, user input processing, and session state management
with graceful startup/shutdown and error handling.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from .config import RAGConfig
from .directory_manager import DirectoryManager
from .commands import CommandRegistry, CommandContext
from .search_service import SearchService

logger = logging.getLogger(__name__)


class InteractiveSession:
    """
    Main interactive session manager for filesystem RAG CLI.
    
    Manages the REPL loop, background monitoring, and graceful shutdown
    while providing a user-friendly command-line interface.
    """
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        self.config = RAGConfig(config_path)
        self.directory_manager = DirectoryManager(self.config)
        self.search_service = SearchService(self.config.weaviate_url, debug=debug)
        self.command_registry = CommandRegistry()
        self.running = False
        self.debug = debug
        
        # Setup logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            self.config.set_setting("debug", True)
    
    async def start(self) -> None:
        """Start the interactive session."""
        try:
            # Print welcome banner
            self._print_banner()
            
            # Load and validate configuration
            await self._initialize_configuration()
            
            # Start directory monitoring
            await self._start_monitoring()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start the main REPL loop
            await self._run_repl()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            logger.error(f"Session error: {e}")
            print(f"Session error: {e}")
        finally:
            await self._cleanup()
    
    def _print_banner(self) -> None:
        """Print the welcome banner."""
        print("🔍 Filesystem RAG Interactive CLI")
        print("=" * 50)
        print("Type /help for available commands or /exit to quit")
        print()
    
    async def _initialize_configuration(self) -> None:
        """Initialize and validate configuration."""
        print("Loading configuration...")
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            print("⚠ Configuration warnings:")
            for error in errors:
                print(f"  - {error}")
            print()
        
        # Show configuration summary
        directories = self.config.list_directories()
        if directories:
            print(f"Found {len(directories)} configured directories")
        else:
            print("No directories configured. Use /add to add directories.")
        
        print(f"Weaviate URL: {self.config.weaviate_url}")
        print()
    
    async def _start_monitoring(self) -> None:
        """Start monitoring configured directories."""
        directories = self.config.list_directories(enabled_only=True)
        
        if not directories:
            print("No directories to monitor.")
            print()
            return
        
        print(f"Starting monitors for {len(directories)} directories...")
        
        try:
            await self.directory_manager.start_all()
            
            # Show monitoring status
            for dir_config in directories:
                monitor = self.directory_manager.get_monitor(dir_config.path)
                if monitor and monitor.is_running:
                    print(f"✓ Monitoring {dir_config.path} ({dir_config.class_prefix})")
                else:
                    print(f"✗ Failed to start monitoring {dir_config.path}")
            
            print()
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            print(f"Warning: Failed to start some monitors: {e}")
            print()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.running = False
            # Schedule cleanup
            asyncio.create_task(self._cleanup())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_repl(self) -> None:
        """Run the main REPL (Read-Eval-Print Loop)."""
        self.running = True
        
        print("Interactive session started. Type /help for commands.")
        print()
        
        # Create command context
        context = CommandContext(
            config=self.config,
            directory_manager=self.directory_manager,
            search_func=self._search_function
        )
        
        while self.running:
            try:
                # Show prompt and get user input
                command_line = await self._get_user_input()
                
                if not command_line.strip():
                    continue
                
                # Execute command
                result = await self.command_registry.execute_command(command_line, context)
                
                # Handle special signals
                if result == "QUIT_SIGNAL":
                    print("Goodbye!")
                    break
                
                # Print result
                if result:
                    print(result)
                    print()
                
            except EOFError:
                # Ctrl+D pressed
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                # Ctrl+C pressed
                print("\nUse /exit to quit gracefully")
                continue
            except Exception as e:
                logger.error(f"REPL error: {e}")
                print(f"Error: {e}")
                print()
    
    async def _get_user_input(self) -> str:
        """Get user input asynchronously."""
        # In a real async environment, we'd use aioconsole or similar
        # For now, use synchronous input
        try:
            return input("> ")
        except EOFError:
            raise
        except KeyboardInterrupt:
            raise
    
    def _search_function(self, query: str, class_prefix: str) -> str:
        """
        Search function for commands using the integrated search service.
        """
        try:
            return self.search_service.search(
                query=query,
                class_prefix=class_prefix,
                temperature=self.config.get_setting("temperature", 0.1),
                search_alpha=self.config.get_setting("search_alpha", 0.8),
                num_results=self.config.get_setting("num_results", 10)
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {e}"
    
    async def _cleanup(self) -> None:
        """Cleanup resources and save state."""
        if not self.running:
            return  # Already cleaned up
        
        print("Saving configuration...")
        
        try:
            # Save configuration
            self.config.save()
            
            # Stop monitoring
            print("Stopping monitors...")
            await self.directory_manager.stop_all()
            
            print("✓ Cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            print(f"Warning: Cleanup error: {e}")
        
        self.running = False
    
    def get_status(self) -> dict:
        """Get current session status."""
        return {
            "running": self.running,
            "config_path": str(self.config.config_path),
            "directories_configured": len(self.config.directories),
            "monitors_running": len([m for m in self.directory_manager.list_monitors() if m.is_running]),
            "weaviate_url": self.config.weaviate_url,
            "debug": self.debug
        }
    
    def __str__(self) -> str:
        status = "✓ Running" if self.running else "✗ Stopped"
        return f"InteractiveSession - {status}"


async def main():
    """Main entry point for the interactive CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filesystem RAG Interactive CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Commands:
  /add <directory> [--class-prefix <prefix>]  Add directory to monitor
  /remove <directory>                         Remove directory from monitoring
  /list                                       Show monitored directories
  /search <query>                             Search indexed content
  /status                                     Show system status
  /help                                       Show all commands
  /exit                                       Exit gracefully

Examples:
  /add /home/user/documents --class-prefix MyDocs
  /search "machine learning configuration"
  /list
        """)
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Configuration file path (default: ~/.filesystem_rag_config.json)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '-w', '--weaviate-url',
        type=str,
        help='Weaviate database URL (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Create and start session
    session = InteractiveSession(
        config_path=args.config,
        debug=args.debug
    )
    
    # Override Weaviate URL if provided
    if args.weaviate_url:
        session.config.weaviate_url = args.weaviate_url
    
    # Start the interactive session
    await session.start()


if __name__ == "__main__":
    asyncio.run(main())