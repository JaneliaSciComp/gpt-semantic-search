"""
Command handlers for interactive filesystem RAG CLI.

Defines all interactive commands available in the CLI interface
with their respective handlers and help text.
"""

import asyncio
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .config import RAGConfig, DirectoryConfig
from .directory_manager import DirectoryManager
from .styled_output import styled_output

logger = logging.getLogger(__name__)


class Command:
    """Base class for CLI commands."""
    
    def __init__(self, name: str, description: str, usage: str = ""):
        self.name = name
        self.description = description
        self.usage = usage
    
    async def execute(self, args: List[str], context: 'CommandContext') -> str:
        """Execute the command with given arguments."""
        raise NotImplementedError
    
    def get_help(self) -> str:
        """Get help text for this command."""
        help_text = f"/{self.name}"
        if self.usage:
            help_text += f" {self.usage}"
        help_text += f"\n  {self.description}"
        return help_text


class CommandContext:
    """Context object passed to command handlers."""
    
    def __init__(
        self,
        config: RAGConfig,
        directory_manager: DirectoryManager,
        search_func: Callable[[str, str], str]
    ):
        self.config = config
        self.directory_manager = directory_manager
        self.search_func = search_func


class AddCommand(Command):
    """Add a directory to monitor."""
    
    def __init__(self):
        super().__init__(
            "add",
            "Add a directory to monitor",
            "<directory> [--class-prefix <prefix>]"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            return "Error: Directory path required\nUsage: /add <directory> [--class-prefix <prefix>]"
        
        directory = args[0]
        class_prefix = None
        
        # Parse optional class prefix
        if "--class-prefix" in args:
            try:
                prefix_idx = args.index("--class-prefix") + 1
                if prefix_idx < len(args):
                    class_prefix = args[prefix_idx]
            except (IndexError, ValueError):
                return "Error: --class-prefix requires a value"
        
        # Use unified class prefix for all directories
        if not class_prefix:
            class_prefix = "Janelia"  # Single unified class for all content
        
        try:
            # Add to configuration
            dir_config = context.config.add_directory(directory, class_prefix)
            
            # Start monitoring
            monitor = await context.directory_manager.add_monitor(dir_config)
            
            # Start background indexing with progress display
            styled_output.print_info(f"Starting background indexing for: {directory}")
            styled_output.print_info(f"Class prefix: {class_prefix}")
            styled_output.print_info("You can continue using the CLI while indexing proceeds in the background.")
            styled_output.console.print()
            
            # Create progress tracking
            import time
            start_time = time.time()
            progress_stats = {
                'files_discovered': 0,
                'files_processed': 0,
                'files_failed': 0,
                'batches_indexed': 0,
                'total_batches': 0,
                'searchable_documents': 0
            }
            
            def progress_callback(stage: str, current: int, total: int):
                """Update progress stats and display."""
                if stage == "searching":
                    progress_stats['files_discovered'] = total
                elif stage == "scraping":
                    progress_stats['files_processed'] = current
                elif stage == "indexing":
                    progress_stats['batches_indexed'] = current
                    progress_stats['total_batches'] = total
                    progress_stats['searchable_documents'] = current * 20  # Estimate based on batch size
                
                # Update display
                styled_output.print_indexing_progress(directory, progress_stats)
            
            try:
                # Use streaming indexing for better performance and user experience
                doc_count = await context.directory_manager.index_directory_streaming(
                    directory,
                    remove_existing=False,
                    progress_callback=progress_callback
                )
                
                # Show completion message
                duration = time.time() - start_time
                styled_output.print_indexing_complete(directory, doc_count, duration)
                
                # Save configuration
                context.config.save()
                
                return ""  # Return empty since styled_output handles the display
                
            except Exception as e:
                styled_output.print_error(f"Indexing failed: {e}")
                styled_output.print_info("✓ Monitoring started (will index files as they change)")
                
                # Save configuration even if indexing failed
                context.config.save()
                
                return ""
            
        except Exception as e:
            return f"Error: {e}"


class RemoveCommand(Command):
    """Remove a directory from monitoring."""
    
    def __init__(self):
        super().__init__(
            "remove",
            "Remove a directory from monitoring",
            "<directory>"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            return "Error: Directory path required\nUsage: /remove <directory>"
        
        directory = args[0]
        
        try:
            # Remove from monitoring
            monitor_removed = await context.directory_manager.remove_monitor(directory)
            
            # Remove from configuration
            config_removed = context.config.remove_directory(directory)
            
            if monitor_removed or config_removed:
                context.config.save()
                return f"✓ Removed directory: {directory}"
            else:
                return f"Directory not found: {directory}"
                
        except Exception as e:
            return f"Error: {e}"


class ListCommand(Command):
    """List all monitored directories."""
    
    def __init__(self):
        super().__init__(
            "list",
            "Show all monitored directories and their status"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        directories = context.config.list_directories()
        
        if not directories:
            return "No directories configured for monitoring."
        
        # Add monitor status to directory configs for display
        for dir_config in directories:
            monitor = context.directory_manager.get_monitor(dir_config.path)
            if monitor:
                status = monitor.get_status()
                dir_config.monitor_status = {
                    'running': status["running"],
                    'files_processed': status['files_processed'],
                    'files_failed': status['files_failed']
                }
            else:
                dir_config.monitor_status = {
                    'running': False,
                    'files_processed': 0,
                    'files_failed': 0
                }
        
        # Use styled output to print the directory list
        styled_output.print_directory_list(directories)
        return ""  # Return empty since styled_output handles the display


class StatusCommand(Command):
    """Show overall monitoring status."""
    
    def __init__(self):
        super().__init__(
            "status",
            "Show monitoring and indexing status"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        status = context.directory_manager.get_status()
        
        # Get directories for status panel
        directories = context.config.list_directories()
        
        # Use styled output to show status
        styled_output.print_status_panel(
            directories, 
            context.config.weaviate_url, 
            status['running_monitors']
        )
        
        return ""  # Return empty since styled_output handles the display


class SearchCommand(Command):
    """Search across all indexed content."""
    
    def __init__(self):
        super().__init__(
            "search",
            "Search across all indexed content",
            "<query>"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            return "Error: Search query required\nUsage: /search <query>"
        
        query = " ".join(args)
        
        # Check if any directories are configured
        directories = context.config.list_directories(enabled_only=True)
        if not directories:
            return "No directories configured for searching."
        
        try:
            # Search in unified Janelia class
            result = context.search_func(query, "Janelia")
            
            if result and "No results found" not in result:
                styled_output.print_search_results(result, query)
                return ""  # Return empty since styled_output handles the display
            else:
                styled_output.print_warning(f"No results found for: '{query}'")
                return ""
                
        except Exception as e:
            return f"Search error: {e}"


class HelpCommand(Command):
    """Show available commands."""
    
    def __init__(self):
        super().__init__(
            "help",
            "Show available commands and usage"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        # Use styled output to show help
        styled_output.print_help()
        return ""  # Return empty since styled_output handles the display


class SettingsCommand(Command):
    """View or update settings."""
    
    def __init__(self):
        super().__init__(
            "settings",
            "View or update configuration settings",
            "[key] [value]"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if len(args) == 0:
            # Show all settings
            result = "Current Settings:\n"
            for key, value in context.config.settings.items():
                result += f"  {key}: {value}\n"
            result += f"\nWeaviate URL: {context.config.weaviate_url}"
            return result
        
        elif len(args) == 1:
            # Show specific setting
            key = args[0]
            if key == "weaviate_url":
                return f"weaviate_url: {context.config.weaviate_url}"
            elif key in context.config.settings:
                return f"{key}: {context.config.settings[key]}"
            else:
                return f"Unknown setting: {key}"
        
        elif len(args) == 2:
            # Update setting
            key, value = args
            
            try:
                # Type conversion
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
                
                if key == "weaviate_url":
                    context.config.weaviate_url = value
                else:
                    context.config.set_setting(key, value)
                
                context.config.save()
                return f"✓ Updated {key} = {value}"
                
            except Exception as e:
                return f"Error updating setting: {e}"
        
        else:
            return "Usage: /settings [key] [value]"


class RestartCommand(Command):
    """Restart monitoring."""
    
    def __init__(self):
        super().__init__(
            "restart",
            "Restart monitoring for a directory or all directories",
            "[directory]"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            # Restart all monitors
            try:
                await context.directory_manager.stop_all()
                await context.directory_manager.start_all()
                return "✓ Restarted all monitors"
            except Exception as e:
                return f"Error restarting monitors: {e}"
        else:
            # Restart specific monitor
            directory = args[0]
            try:
                success = await context.directory_manager.restart_monitor(directory)
                if success:
                    return f"✓ Restarted monitor for: {directory}"
                else:
                    return f"Monitor not found: {directory}"
            except Exception as e:
                return f"Error restarting monitor: {e}"


class IndexCommand(Command):
    """Force re-indexing of a directory."""
    
    def __init__(self):
        super().__init__(
            "index",
            "Force re-indexing of a directory",
            "<directory> [--remove-existing]"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            return "Error: Directory path required\nUsage: /index <directory> [--remove-existing]"
        
        directory = args[0]
        remove_existing = "--remove-existing" in args
        
        try:
            def progress_callback(current: int, total: int):
                if current == 0:
                    styled_output.print_info(f"Re-indexing {total} files...")
                elif current == total:
                    styled_output.print_success("Re-indexing complete")
            
            doc_count = await context.directory_manager.index_directory(
                directory,
                remove_existing=remove_existing,
                progress_callback=progress_callback
            )
            
            return f"✓ Re-indexed {doc_count} files from: {directory}"
            
        except Exception as e:
            return f"Error re-indexing directory: {e}"


class AgentCommand(Command):
    """Chat with the filesystem RAG agent."""
    
    def __init__(self):
        super().__init__(
            "agent",
            "Chat with the filesystem RAG agent",
            "<message>"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        if not args:
            return "Error: Message required\nUsage: /agent <message>"
        
        message = " ".join(args)
        
        try:
            # Ensure agent service is initialized (lazy initialization)
            if hasattr(context, 'ensure_agent_service') and not context.agent_service:
                agent_available = await context.ensure_agent_service()
                if not agent_available:
                    return "Agent service initialization failed. Check configuration and server status."
                # Update context with initialized service
                context.agent_service = context.ensure_agent_service.__self__.agent_service
            elif not hasattr(context, 'agent_service') or not context.agent_service:
                return "Agent service not available. Check configuration and server status."
            
            # Query the agent
            response = context.agent_service.query(message)
            
            # Use styled output for agent responses
            styled_output.print_agent_response(response, message)
            return ""  # Return empty since styled_output handles the display
            
        except Exception as e:
            return f"Agent error: {e}"


class AgentStatusCommand(Command):
    """Show agent status and configuration."""
    
    def __init__(self):
        super().__init__(
            "agent-status",
            "Show agent status and configuration"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        try:
            # Try lazy initialization first
            if hasattr(context, 'ensure_agent_service') and not context.agent_service:
                await context.ensure_agent_service()
                context.agent_service = context.ensure_agent_service.__self__.agent_service
                
            if not hasattr(context, 'agent_service') or not context.agent_service:
                return "Agent service not configured."
            
            status = context.agent_service.get_status()
            
            result = "🤖 Agent Status:\n\n"
            result += f"• Initialized: {'✅' if status['initialized'] else '❌'}\n"
            result += f"• Using Local LLM: {'✅' if status['using_local_llm'] else '❌ (disabled)'}\n"
            
            if 'error' in status:
                result += f"• Error: {status['error']}\n"
            
            if 'local_llm_status' in status:
                llm_status = status['local_llm_status']
                result += f"\n🖥️ Local LLM Server:\n"
                result += f"• URL: {llm_status['server_url']}\n"
                result += f"• Connected: {'✅' if llm_status['connected'] else '❌'}\n"
                result += f"• LLM Initialized: {'✅' if llm_status['llm_initialized'] else '❌'}\n"
            
            config = status['config']
            result += f"\n⚙️ Configuration:\n"
            result += f"• Temperature: {config['temperature']}\n"
            result += f"• Max Tokens: {config['max_tokens']}\n"
            result += f"• Server Port: {config['server_port']}\n"
            result += f"• Ollama Integration: {'✅' if config.get('ollama_enabled', True) else '❌'}\n"
            
            if 'tools_available' in status:
                result += f"\n🔧 Tools Available: {status['tools_available']}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting agent status: {e}"


class AgentHelpCommand(Command):
    """Show agent help and example queries."""
    
    def __init__(self):
        super().__init__(
            "agent-help",
            "Show agent help and example queries"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        try:
            # Try lazy initialization first
            if hasattr(context, 'ensure_agent_service') and not context.agent_service:
                await context.ensure_agent_service()
                context.agent_service = context.ensure_agent_service.__self__.agent_service
                
            if not hasattr(context, 'agent_service') or not context.agent_service:
                help_text = """
🤖 Filesystem RAG Agent (Not Available)

The agent service is not configured. To enable the agent:

1. Install dependencies: pixi install
2. Start llama-server (see agent-server command)
3. Use /agent command to initialize

Basic agent commands:
• /agent <message>     - Chat with the agent (will auto-initialize)
• /agent-status        - Show agent status
• /agent-server        - Show server command
• /agent-help          - Show this help
"""
                return help_text
            
            return context.agent_service.get_help_text()
            
        except Exception as e:
            return f"Error getting agent help: {e}"


class AgentServerCommand(Command):
    """Show llama-server startup command."""
    
    def __init__(self):
        super().__init__(
            "agent-server",
            "Show the llama-server startup command"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        try:
            # Try lazy initialization first
            if hasattr(context, 'ensure_agent_service') and not context.agent_service:
                await context.ensure_agent_service()
                context.agent_service = context.ensure_agent_service.__self__.agent_service
                
            if not hasattr(context, 'agent_service') or not context.agent_service:
                return "Agent service not configured."
            
            server_command = context.agent_service.get_server_command()
            
            if not server_command:
                return "Agent not configured to use local LLM."
            
            result = "🖥️ Llama Server Startup Command:\n\n"
            result += f"```\n{server_command}\n```\n\n"
            result += "📋 Instructions:\n"
            result += "1. Copy and run the above command in a separate terminal\n"
            result += "2. Wait for the server to start (shows 'HTTP server listening')\n"
            result += "3. Use /agent-status to verify connection\n"
            result += "4. Start chatting with /agent <your message>\n\n"
            result += "💡 Tips:\n"
            result += "• Make sure your-model.gguf exists or specify full path\n"
            result += "• The --chat-template chatml-function-calling is required for tool support\n"
            result += "• Server runs on port 8080 by default\n"
            
            return result
            
        except Exception as e:
            return f"Error getting server command: {e}"


class SyncCommand(Command):
    """Sync filesystem with Weaviate database."""
    
    def __init__(self):
        super().__init__(
            "sync",
            "Sync filesystem with Weaviate database to detect changes"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        try:
            styled_output.print_info("🔄 Starting filesystem-database sync...")
            
            # Perform the sync
            sync_results = await context.directory_manager.sync_filesystem_with_database()
            
            # Format results
            result = "📊 Sync Results:\n\n"
            result += f"• Files added: {sync_results['files_added']}\n"
            result += f"• Files updated: {sync_results['files_updated']}\n"
            result += f"• Files removed: {sync_results['files_removed']}\n"
            result += f"• Files unchanged: {sync_results['files_unchanged']}\n"
            
            if sync_results['errors']:
                result += f"\n❌ Errors ({len(sync_results['errors'])}):\n"
                for error in sync_results['errors'][:5]:  # Show first 5 errors
                    result += f"  • {error}\n"
                if len(sync_results['errors']) > 5:
                    result += f"  ... and {len(sync_results['errors']) - 5} more errors\n"
            
            return result
            
        except Exception as e:
            return f"Sync failed: {e}"


class ExitCommand(Command):
    """Exit the CLI."""
    
    def __init__(self):
        super().__init__(
            "exit",
            "Exit the CLI gracefully"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        return "QUIT_SIGNAL"  # Special signal to exit


class CommandRegistry:
    """Registry of available CLI commands."""
    
    def __init__(self):
        self.commands: Dict[str, Command] = {}
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register all default commands."""
        commands = [
            AddCommand(),
            RemoveCommand(),
            ListCommand(),
            StatusCommand(),
            SearchCommand(),
            HelpCommand(),
            SettingsCommand(),
            RestartCommand(),
            IndexCommand(),
            SyncCommand(),
            AgentCommand(),
            AgentStatusCommand(),
            AgentHelpCommand(),
            AgentServerCommand(),
            ExitCommand()
        ]
        
        for command in commands:
            self.register(command)
            
        # Add aliases and shortcuts
        self.commands["quit"] = self.commands["exit"]
        self.commands["q"] = self.commands["exit"]
        self.commands["ls"] = self.commands["list"]
        self.commands["l"] = self.commands["list"]
        self.commands["?"] = self.commands["help"]
        self.commands["h"] = self.commands["help"]
        self.commands["s"] = self.commands["status"]
        self.commands["a"] = self.commands["add"]
        self.commands["r"] = self.commands["remove"]
        # Agent aliases
        self.commands["ai"] = self.commands["agent"]
        self.commands["chat"] = self.commands["agent"]
    
    def register(self, command: Command):
        """Register a new command."""
        self.commands[command.name] = command
    
    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name."""
        return self.commands.get(name.lower())
    
    def list_commands(self) -> List[str]:
        """Get list of all command names."""
        return sorted(self.commands.keys())
    
    async def execute_command(
        self,
        command_line: str,
        context: CommandContext
    ) -> str:
        """
        Parse and execute a command line.
        
        Args:
            command_line: Full command line (e.g., "/add /path/to/dir")
            context: Command context
            
        Returns:
            Command result string
        """
        if not command_line.startswith("/"):
            return "Commands must start with '/'. Type naturally to search, or /help for available commands."
        
        # Parse command and arguments
        parts = command_line[1:].split()
        if not parts:
            return "Type /help for available commands."
        
        command_name = parts[0]
        args = parts[1:]
        
        command = self.get_command(command_name)
        if not command:
            return f"Unknown command: /{command_name}. Type /help for available commands."
        
        try:
            return await command.execute(args, context)
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return f"Command error: {e}"