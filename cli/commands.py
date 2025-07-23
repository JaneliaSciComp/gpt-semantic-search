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
        
        # Generate class prefix if not provided
        if not class_prefix:
            dir_name = Path(directory).name
            class_prefix = re.sub(r'[^a-zA-Z0-9]', '', dir_name.title())
            if not class_prefix:
                class_prefix = "Directory"
        
        try:
            # Add to configuration
            dir_config = context.config.add_directory(directory, class_prefix)
            
            # Start monitoring
            monitor = await context.directory_manager.add_monitor(dir_config)
            
            # Perform initial indexing
            result = f"Adding directory: {directory}\n"
            result += f"Class prefix: {class_prefix}\n"
            
            try:
                def progress_callback(current: int, total: int):
                    if current == 0:
                        print(f"Indexing {total} files...")
                    elif current == total:
                        print(f"✓ Indexing complete")
                
                doc_count = await context.directory_manager.index_directory(
                    directory,
                    remove_existing=False,
                    progress_callback=progress_callback
                )
                
                result += f"✓ Indexed {doc_count} files\n"
                result += f"✓ Monitoring started"
                
            except Exception as e:
                result += f"⚠ Indexing failed: {e}\n"
                result += f"✓ Monitoring started (will index files as they change)"
            
            # Save configuration
            context.config.save()
            
            return result
            
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
        
        result = "Monitored Directories:\n"
        
        for dir_config in directories:
            monitor = context.directory_manager.get_monitor(dir_config.path)
            
            if monitor:
                status = monitor.get_status()
                running_status = "✓ Active" if status["running"] else "✗ Stopped"
                files_info = f"{status['files_processed']} processed"
                if status['files_failed'] > 0:
                    files_info += f", {status['files_failed']} failed"
                
                # Count files in directory
                try:
                    total_files = len([f for f in Path(dir_config.path).rglob("*") if f.is_file()])
                    files_info = f"{total_files} files - {files_info}"
                except:
                    pass
                
                result += f"• {dir_config.path} ({dir_config.class_prefix}) - {files_info} - {running_status}\n"
            else:
                status = "✗ Disabled" if not dir_config.enabled else "✗ Not Running"
                result += f"• {dir_config.path} ({dir_config.class_prefix}) - {status}\n"
        
        return result.rstrip()


class StatusCommand(Command):
    """Show overall monitoring status."""
    
    def __init__(self):
        super().__init__(
            "status",
            "Show monitoring and indexing status"
        )
    
    async def execute(self, args: List[str], context: CommandContext) -> str:
        status = context.directory_manager.get_status()
        
        result = "🔍 Filesystem RAG Status\n"
        result += f"Weaviate URL: {context.config.weaviate_url}\n"
        result += f"Total Monitors: {status['total_monitors']}\n"
        result += f"Running Monitors: {status['running_monitors']}\n"
        result += f"Files Processed: {status['total_files_processed']}\n"
        
        if status['total_files_failed'] > 0:
            result += f"Files Failed: {status['total_files_failed']}\n"
        
        result += f"Configuration: {context.config.config_path}\n"
        
        return result


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
        
        # Get all class prefixes from configured directories
        class_prefixes = [d.class_prefix for d in context.config.list_directories(enabled_only=True)]
        
        if not class_prefixes:
            return "No directories configured for searching."
        
        try:
            # Search across all classes
            results = []
            for class_prefix in class_prefixes:
                try:
                    result = context.search_func(query, class_prefix)
                    if result and "No results found" not in result:
                        results.append(f"Results from {class_prefix}:\n{result}")
                except Exception as e:
                    logger.error(f"Search failed for {class_prefix}: {e}")
                    continue
            
            if results:
                return "\n\n" + "="*50 + "\n\n".join(results)
            else:
                return f"No results found for: '{query}'"
                
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
        return """
🔍 Filesystem RAG Interactive CLI Commands:

/add <directory> [--class-prefix <prefix>]
  Add a directory to monitor and index

/remove <directory>
  Remove a directory from monitoring

/list
  Show all monitored directories and their status

/status
  Show overall monitoring and indexing status

/search <query>
  Search across all indexed content

/settings [key] [value]
  View or update configuration settings

/restart [directory]
  Restart monitoring for a specific directory (or all)

/index <directory>
  Force re-indexing of a directory

/help
  Show this help message

/exit or /quit
  Exit the CLI gracefully

Examples:
  /add /home/user/documents --class-prefix MyDocs
  /search "machine learning configuration"
  /settings debug true
  /restart /home/user/documents
        """.strip()


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
                    print(f"Re-indexing {total} files...")
                elif current == total:
                    print(f"✓ Re-indexing complete")
            
            doc_count = await context.directory_manager.index_directory(
                directory,
                remove_existing=remove_existing,
                progress_callback=progress_callback
            )
            
            return f"✓ Re-indexed {doc_count} files from: {directory}"
            
        except Exception as e:
            return f"Error re-indexing directory: {e}"


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
            ExitCommand()
        ]
        
        for command in commands:
            self.register(command)
            
        # Add aliases
        self.commands["quit"] = self.commands["exit"]
        self.commands["ls"] = self.commands["list"]
        self.commands["?"] = self.commands["help"]
    
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
            return "Commands must start with '/'. Type /help for available commands."
        
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