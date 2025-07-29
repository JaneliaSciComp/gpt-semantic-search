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

from rich.console import Console
from rich.prompt import Prompt

from .config import RAGConfig
from .directory_manager import DirectoryManager
from .commands import CommandRegistry, CommandContext
from .search_service import SearchService
from .styled_output import styled_output
from .agent_service import AgentService
from .agent_config import get_default_config

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
        self.console = Console()
        
        # Initialize agent service
        self.agent_service: Optional[AgentService] = None
        
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
        styled_output.print_banner()
    
    async def _initialize_configuration(self) -> None:
        """Initialize and validate configuration."""
        with styled_output.create_status_spinner("Loading configuration..."):
            await asyncio.sleep(0.1)  # Brief pause for visual effect
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            styled_output.print_warning("Configuration warnings:")
            for error in errors:
                styled_output.print_info(f"  - {error}")
        
        # Show configuration summary
        directories = self.config.list_directories()
        if not directories:
            styled_output.print_info("No directories configured. Use /add to add directories.")
        
        # Show status panel
        monitors_running = len([d for d in directories if d.enabled])
        styled_output.print_status_panel(directories, self.config.weaviate_url, monitors_running)
    
    async def _ensure_agent_service(self) -> bool:
        """Ensure agent service is initialized (lazy initialization)."""
        if self.agent_service is not None:
            return True
            
        try:
            with styled_output.create_status_spinner("Initializing agent service..."):
                # Get monitored directories for tools
                directories = self.config.list_directories()
                monitored_dirs = {d.path: d.class_prefix for d in directories if d.enabled}
                
                # Initialize agent service
                self.agent_service = AgentService(
                    search_service=self.search_service,
                    config=get_default_config(),
                    monitored_directories=monitored_dirs
                )
                
                # Try to initialize (will fallback to OpenAI if local LLM unavailable)
                initialized = self.agent_service.initialize()
                
                if initialized:
                    if self.agent_service.using_local_llm:
                        styled_output.print_success("✓ Agent service initialized with local LLM")
                    else:
                        styled_output.print_info("✓ Agent service initialized with OpenAI fallback")
                    return True
                else:
                    styled_output.print_warning("⚠ Agent service initialization failed")
                    if self.agent_service.initialization_error:
                        styled_output.print_info(f"  Error: {self.agent_service.initialization_error}")
                    self.agent_service = None
                    return False
                
        except Exception as e:
            logger.error(f"Agent service initialization failed: {e}")
            styled_output.print_warning(f"⚠ Agent service unavailable: {e}")
            self.agent_service = None
            return False
    
    async def _start_monitoring(self) -> None:
        """Start monitoring configured directories."""
        directories = self.config.list_directories(enabled_only=True)
        
        if not directories:
            styled_output.print_info("No directories to monitor.")
            return
        
        with styled_output.create_status_spinner(f"Starting monitors for {len(directories)} directories..."):
            try:
                await self.directory_manager.start_all()
                await asyncio.sleep(0.5)  # Brief pause for visual effect
            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                styled_output.print_warning(f"Failed to start some monitors: {e}")
                return
        
        # Show monitoring status
        success_count = 0
        for dir_config in directories:
            monitor = self.directory_manager.get_monitor(dir_config.path)
            if monitor and monitor.is_running:
                styled_output.print_success(f"Monitoring {dir_config.path} ({dir_config.class_prefix})")
                success_count += 1
            else:
                styled_output.print_error(f"Failed to start monitoring {dir_config.path}")
        
        if success_count > 0:
            styled_output.print_info(f"Started {success_count}/{len(directories)} monitors successfully")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.running = False
            # Schedule cleanup
            asyncio.create_task(self._cleanup())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_repl(self) -> None:
        """Run the main REPL (Read-Eval-Print Loop)."""
        self.running = True
        
        styled_output.print_info("Interactive session started. Type naturally to search, or /help for commands.")
        styled_output.console.print()
        
        # Create command context
        context = CommandContext(
            config=self.config,
            directory_manager=self.directory_manager,
            search_func=self._search_function
        )
        
        # Add agent service and lazy initialization function to context
        context.agent_service = self.agent_service
        context.ensure_agent_service = self._ensure_agent_service
        
        while self.running:
            try:
                # Show prompt and get user input
                user_input = await self._get_user_input()
                
                if not user_input.strip():
                    continue
                
                # Determine if this is a command or search query
                if user_input.startswith("/"):
                    # This is a command
                    result = await self.command_registry.execute_command(user_input, context)
                    
                    # Handle special signals
                    if result == "QUIT_SIGNAL":
                        styled_output.print_goodbye()
                        break
                    
                    # Print command result
                    if result:
                        styled_output.print_command_result(result)
                        styled_output.console.print()
                elif user_input.lower() in ['help', 'status', 'list', 'exit', 'quit']:
                    # Common commands that can be used without /
                    result = await self.command_registry.execute_command(f"/{user_input.lower()}", context)
                    
                    # Handle special signals
                    if result == "QUIT_SIGNAL":
                        styled_output.print_goodbye()
                        break
                    
                    # Print command result
                    if result:
                        styled_output.print_command_result(result)
                        styled_output.console.print()
                else:
                    # This is a search query - execute search directly
                    result = await self._handle_search_query(user_input, context)
                    if result:
                        styled_output.print_search_results(result, user_input)
                
            except EOFError:
                # Ctrl+D pressed
                styled_output.print_goodbye()
                break
            except KeyboardInterrupt:
                # Ctrl+C pressed
                styled_output.print_warning("Use /exit to quit gracefully")
                continue
            except Exception as e:
                logger.error(f"REPL error: {e}")
                styled_output.print_error(f"Error: {e}")
    
    async def _get_user_input(self) -> str:
        """Get user input with styled prompt."""
        # Get current status for prompt context
        directories = self.config.list_directories()
        monitors_running = len([d for d in directories if d.enabled and 
                              self.directory_manager.get_monitor(d.path) and 
                              self.directory_manager.get_monitor(d.path).is_running])
        
        # Generate styled prompt
        prompt_text = styled_output.get_styled_prompt(
            directories_count=len(directories),
            monitors_active=monitors_running
        )
        
        try:
            return Prompt.ask(prompt_text, console=styled_output.console, default="", show_default=False)
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
                num_results=self.config.get_setting("num_results", 3)
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {e}"
    
    async def _handle_search_query(self, query: str, context: CommandContext) -> str:
        """
        Handle natural language search queries (non-command input).
        """
        # Get all class prefixes from configured directories
        class_prefixes = [d.class_prefix for d in self.config.list_directories(enabled_only=True)]
        
        if not class_prefixes:
            return "No directories configured for searching. Use /add to add directories."
        
        try:
            # Search across all classes
            results = []
            for class_prefix in class_prefixes:
                try:
                    result = self._search_function(query, class_prefix)
                    if result and "No results found" not in result and "Search failed" not in result:
                        results.append(f"**Results from {class_prefix}:**\n{result}")
                except Exception as e:
                    logger.error(f"Search failed for {class_prefix}: {e}")
                    continue
            
            if results:
                return "\n\n" + "---\n\n".join(results)
            else:
                return ""  # Will be handled as "no results" by caller
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {e}"
    
    async def _cleanup(self) -> None:
        """Cleanup resources and save state."""
        if not self.running:
            return  # Already cleaned up
        
        with styled_output.create_status_spinner("Saving configuration and stopping monitors..."):
            try:
                # Save configuration
                self.config.save()
                
                # Stop monitoring
                await self.directory_manager.stop_all()
                
                # Shutdown agent service
                if self.agent_service:
                    self.agent_service.shutdown()
                
                await asyncio.sleep(0.3)  # Brief pause for visual effect
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                styled_output.print_warning(f"Cleanup error: {e}")
        
        styled_output.print_success("Cleanup complete")
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