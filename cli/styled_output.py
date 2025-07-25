"""
Styled output module for filesystem RAG CLI using Rich library.

Provides beautiful, modern terminal output with colors, formatting, and interactive elements
to create a Claude Code-like user experience.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich import box
from rich.status import Status


class StyledOutput:
    """
    Handles all styled output for the CLI using Rich library.
    
    Provides methods for consistent, beautiful terminal output with colors,
    formatting, and interactive elements.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.theme_colors = {
            'primary': '#00D9FF',      # Bright cyan - main brand color
            'secondary': '#7C3AED',    # Purple - secondary actions
            'success': '#10B981',      # Green - success states
            'warning': '#F59E0B',      # Amber - warnings
            'error': '#EF4444',        # Red - errors
            'muted': '#6B7280',        # Gray - secondary text
            'text': '#F9FAFB',         # Light gray - primary text
            'accent': '#8B5CF6'        # Light purple - accents
        }
    
    def print_banner(self) -> None:
        """Print the welcome banner with styling."""
        banner_text = Text()
        banner_text.append("Filesystem RAG Interactive CLI", style="bold bright_cyan")
        
        panel = Panel(
            Align.center(banner_text),
            box=box.ROUNDED,
            border_style="bright_cyan",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()
        
        # Quick help hint
        help_text = Text()
        help_text.append("Type naturally to search, or use ", style="dim")
        help_text.append("/help", style="bright_cyan")
        help_text.append(" for commands", style="dim")
        
        self.console.print(Align.center(help_text))
        self.console.print()
    
    def print_status_panel(self, directories: List[Dict], weaviate_url: str, 
                          monitors_running: int) -> None:
        """Print system status in a formatted panel."""
        
        # Create status table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", style="white")
        
        table.add_row("Weaviate", weaviate_url)
        table.add_row("Directories", str(len(directories)))
        table.add_row("Active Monitors", f"[green]{monitors_running}[/]" if monitors_running > 0 else "[dim]0[/]")
        
        if directories:
            table.add_row("", "")  # Spacer
            for dir_config in directories[:3]:  # Show first 3
                # Handle both DirectoryConfig objects and dictionaries
                if hasattr(dir_config, 'enabled'):
                    status_icon = "[green]•[/]" if dir_config.enabled else "[dim]•[/]"
                    class_prefix = dir_config.class_prefix
                    path = dir_config.path
                else:
                    status_icon = "[green]•[/]" if dir_config.get('enabled', True) else "[dim]•[/]"
                    class_prefix = dir_config.get('class_prefix', 'Unknown')
                    path = dir_config.get('path', 'Unknown path')
                
                table.add_row(
                    f"{status_icon} {class_prefix}",
                    f"[dim]{path}[/]"
                )
            
            if len(directories) > 3:
                table.add_row("[dim]...[/]", f"[dim]+{len(directories) - 3} more[/]")
        
        panel = Panel(
            table,
            title="[bold bright_cyan]System Status[/]",
            border_style="bright_blue",
            box=box.ROUNDED
        )
        
        self.console.print(panel)
        self.console.print()
    
    def print_search_results(self, results: str, query: str) -> None:
        """Format and print search results."""
        if not results or "No results found" in results:
            self.print_warning(f"No results found for: '{query}'")
            return
        
        # Create a panel for search results
        panel = Panel(
            Markdown(results),
            title=f"[bold bright_cyan]Search Results[/] [dim]'{query}'[/]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()
    
    def print_directory_list(self, directories) -> None:
        """Print formatted list of monitored directories."""
        if not directories:
            self.print_info("No directories configured for monitoring.")
            return
        
        table = Table(
            title="[bold bright_cyan]Monitored Directories[/]",
            box=box.ROUNDED,
            header_style="bold bright_blue"
        )
        
        table.add_column("Status", width=8)
        table.add_column("Directory", style="white", no_wrap=False)
        table.add_column("Class Prefix", style="bright_cyan")
        table.add_column("Files", justify="right", style="dim")
        
        for dir_config in directories:
            # Handle both DirectoryConfig objects and dictionaries
            if hasattr(dir_config, 'enabled'):
                # DirectoryConfig object
                enabled = dir_config.enabled
                path = dir_config.path
                class_prefix = dir_config.class_prefix
                monitor_status = getattr(dir_config, 'monitor_status', {})
            else:
                # Dictionary
                enabled = dir_config.get('enabled', True)
                path = dir_config.get('path', 'Unknown')
                class_prefix = dir_config.get('class_prefix', 'Unknown')
                monitor_status = dir_config.get('monitor_status', {})
            
            # Determine status
            if monitor_status.get('running', False):
                status = "[green]Active[/]"
            elif enabled:
                status = "[yellow]Stopped[/]"
            else:
                status = "[dim]Disabled[/]"
            
            # File counts
            files_processed = monitor_status.get('files_processed', 0)
            files_failed = monitor_status.get('files_failed', 0)
            
            files_info = str(files_processed)
            if files_failed > 0:
                files_info += f" [red]({files_failed} failed)[/]"
            
            table.add_row(
                status,
                path,
                class_prefix,
                files_info
            )
        
        self.console.print(table)
        self.console.print()
    
    def print_success(self, message: str) -> None:
        """Print success message with green styling."""
        self.console.print(f"[green]{message}[/]")
    
    def print_error(self, message: str) -> None:
        """Print error message with red styling."""
        self.console.print(f"[red]Error: {message}[/]")
    
    def print_warning(self, message: str) -> None:
        """Print warning message with yellow styling."""
        self.console.print(f"[yellow]Warning: {message}[/]")
    
    def print_info(self, message: str) -> None:
        """Print info message with blue styling."""
        self.console.print(f"[bright_blue]{message}[/]")
    
    def print_help(self) -> None:
        """Print formatted help information."""
        help_content = """
# Filesystem RAG Interactive CLI

## Search Mode (Default)
Just type your query naturally - no commands needed!

## Available Commands

### Directory Management
- `/add <directory> [--class-prefix <prefix>]` - Add directory to monitor
- `/remove <directory>` - Remove directory from monitoring  
- `/list` - Show all monitored directories
- `/restart [directory]` - Restart monitoring

### Search & Information
- `/search <query>` - Explicit search (same as typing naturally)
- `/status` - Show system status
- `/settings [key] [value]` - View/update settings

### System Commands  
- `/index <directory>` - Force re-index directory
- `/help` - Show this help
- `/exit` or `/quit` - Exit gracefully

## Quick Examples
```
How do I configure the system?
/add ~/Documents --class-prefix MyDocs  
/status
machine learning best practices
/settings debug true
```

## Tips
- Common commands work without `/` prefix (help, status, list, exit)
- Use `/restart` if monitoring stops working
- Check `status` to see what's being monitored
- Search works across all your indexed directories
- Just type naturally to search - no `/search` needed
        """
        
        panel = Panel(
            Markdown(help_content),
            title="[bold bright_cyan]Help[/]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def create_progress_bar(self, description: str = "Processing") -> Progress:
        """Create a styled progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
    
    def create_status_spinner(self, message: str) -> Status:
        """Create a status spinner for long operations."""
        return Status(
            message,
            console=self.console,
            spinner="dots"
        )
    
    def get_styled_prompt(self, directories_count: int = 0, 
                         monitors_active: int = 0) -> str:
        """Generate a styled prompt with context indicators."""
        
        # Build context indicators
        indicators = []
        
        if directories_count > 0:
            indicators.append(f"[dim]{directories_count} dirs[/]")
        
        if monitors_active > 0:
            indicators.append(f"[green]{monitors_active} active[/]")
        elif directories_count > 0:
            indicators.append("[yellow]monitoring off[/]")
        
        # Create prompt parts
        prompt_parts = []
        
        if indicators:
            context = " | ".join(indicators)
            prompt_parts.append(f"[dim]({context})[/]")
        
        prompt_parts.append("[bright_cyan]>[/]")
        
        return " ".join(prompt_parts) + " "
    
    def print_command_result(self, result: str, command: str = "") -> None:
        """Print command result with appropriate styling."""
        if not result:
            return
        
        # Detect result type and style accordingly
        if result.startswith("✓") or "success" in result.lower():
            # Success results get a subtle green tint
            styled_text = Text(result)
            styled_text.stylize("green", 0, 1)  # Just the checkmark
            self.console.print(styled_text)
        elif result.startswith("Error:") or result.startswith("❌"):
            self.console.print(f"[red]{result}[/]")
        elif result.startswith("Warning:") or result.startswith("⚠"):
            self.console.print(f"[yellow]{result}[/]")
        elif "Results from" in result:
            # This is search results - use markdown formatting
            self.console.print(Markdown(result))
        else:
            # Regular output
            self.console.print(result)
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_agent_response(self, response: str, query: str = "") -> None:
        """Print agent response with special formatting."""
        # Create header with robot emoji
        if query:
            header = f"🤖 Agent Response to: {query}"
        else:
            header = "🤖 Agent Response"
        
        # Create styled panel for the response
        response_panel = Panel(
            Markdown(response),
            title=f"[bold bright_cyan]{header}[/]",
            title_align="left",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(response_panel)
    
    def print_goodbye(self) -> None:
        """Print goodbye message."""
        goodbye_text = Text()
        goodbye_text.append("Thanks for using Filesystem RAG CLI!", style="bright_cyan")
        
        self.console.print()
        self.console.print(Align.center(goodbye_text))
        self.console.print()


# Global instance for easy access
styled_output = StyledOutput()