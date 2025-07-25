"""
Agent tools for filesystem RAG operations.

Provides tools that the FunctionAgent can use to search, list,
and interact with the indexed filesystem content.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.tools import FunctionTool

from .search_service import SearchService

logger = logging.getLogger(__name__)


class FilesystemAgentTools:
    """Tools for the filesystem RAG agent."""
    
    def __init__(self, search_service: SearchService, monitored_directories: Optional[Dict[str, str]] = None):
        self.search_service = search_service
        self.monitored_directories = monitored_directories or {}
    
    def search_indexed_content(self, query: str, class_prefix: Optional[str] = None) -> str:
        """
        Search indexed filesystem content semantically.
        
        Args:
            query: The search query to execute
            class_prefix: Optional specific class prefix to search (if None, searches all)
            
        Returns:
            Formatted search results with sources
        """
        try:
            if class_prefix:
                # Search specific class
                if not self.search_service.check_class_exists(class_prefix):
                    return f"Class '{class_prefix}' does not exist in the index."
                
                return self.search_service.search(
                    query=query,
                    class_prefix=class_prefix,
                    temperature=0.1,
                    search_alpha=0.8,
                    num_results=5
                )
            else:
                # Search all available classes
                available_classes = self.search_service.list_classes()
                if not available_classes:
                    return "No indexed content available. Please add directories first."
                
                return self.search_service.search_multiple_classes(
                    query=query,
                    class_prefixes=available_classes,
                    temperature=0.1,
                    search_alpha=0.8,
                    num_results=3
                )
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {e}"
    
    def list_indexed_classes(self) -> str:
        """
        List all available indexed classes in Weaviate.
        
        Returns:
            Formatted list of available classes with document counts
        """
        try:
            classes = self.search_service.list_classes()
            if not classes:
                return "No indexed classes found. Add directories to start indexing."
            
            result = "Available indexed classes:\n\n"
            for class_prefix in classes:
                info = self.search_service.get_class_info(class_prefix)
                if info['exists']:
                    count = info['document_count']
                    result += f"• {class_prefix}: {count} documents\n"
                else:
                    result += f"• {class_prefix}: Error - {info.get('error', 'Unknown error')}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list classes: {e}")
            return f"Failed to list classes: {e}"
    
    def get_class_details(self, class_prefix: str) -> str:
        """
        Get detailed information about a specific indexed class.
        
        Args:
            class_prefix: The class prefix to get details for
            
        Returns:
            Detailed information about the class
        """
        try:
            info = self.search_service.get_class_info(class_prefix)
            
            if not info['exists']:
                return f"Class '{class_prefix}' does not exist."
            
            result = f"Class Details: {class_prefix}\n\n"
            result += f"• Full class name: {info['class_name']}\n"
            result += f"• Document count: {info['document_count']}\n"
            result += f"• Properties: {', '.join(info['properties'])}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get class details: {e}")
            return f"Failed to get class details: {e}"
    
    def list_directory_contents(self, directory_path: str, max_items: int = 20) -> str:
        """
        List contents of a directory on the filesystem.
        
        Args:
            directory_path: Path to the directory to list
            max_items: Maximum number of items to return
            
        Returns:
            Formatted directory listing
        """
        try:
            path = Path(directory_path).expanduser().resolve()
            
            if not path.exists():
                return f"Directory does not exist: {directory_path}"
            
            if not path.is_dir():
                return f"Path is not a directory: {directory_path}"
            
            items = []
            try:
                for item in path.iterdir():
                    if len(items) >= max_items:
                        break
                    
                    item_type = "📁" if item.is_dir() else "📄"
                    size = ""
                    if item.is_file():
                        try:
                            size_bytes = item.stat().st_size
                            if size_bytes < 1024:
                                size = f" ({size_bytes}B)"
                            elif size_bytes < 1024 * 1024:
                                size = f" ({size_bytes/1024:.1f}KB)"
                            else:
                                size = f" ({size_bytes/(1024*1024):.1f}MB)"
                        except:
                            pass
                    
                    items.append(f"{item_type} {item.name}{size}")
                
            except PermissionError:
                return f"Permission denied accessing: {directory_path}"
            
            result = f"Directory contents: {directory_path}\n\n"
            if items:
                result += "\n".join(items)
                if len(list(path.iterdir())) > max_items:
                    result += f"\n\n... and {len(list(path.iterdir())) - max_items} more items"
            else:
                result += "Directory is empty"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list directory: {e}")
            return f"Failed to list directory: {e}"
    
    def read_file_content(self, file_path: str, max_chars: int = 2000) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path: Path to the file to read
            max_chars: Maximum number of characters to return
            
        Returns:
            File content (truncated if necessary)
        """
        try:
            path = Path(file_path).expanduser().resolve()
            
            if not path.exists():
                return f"File does not exist: {file_path}"
            
            if not path.is_file():
                return f"Path is not a file: {file_path}"
            
            # Check file size
            try:
                size = path.stat().st_size
                if size > 10 * 1024 * 1024:  # 10MB
                    return f"File too large to read: {file_path} ({size} bytes)"
            except:
                pass
            
            # Try to read as text
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = path.read_text(encoding='latin-1')
                except:
                    return f"Cannot read file as text: {file_path}"
            
            # Truncate if necessary
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... (truncated, showing first {max_chars} characters)"
            
            return f"File content: {file_path}\n\n{content}"
            
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return f"Failed to read file: {e}"
    
    def get_monitored_directories(self) -> str:
        """
        Get list of currently monitored directories.
        
        Returns:
            Formatted list of monitored directories
        """
        if not self.monitored_directories:
            return "No directories are currently being monitored."
        
        result = "Monitored directories:\n\n"
        for directory, class_prefix in self.monitored_directories.items():
            result += f"• {directory} → Class: {class_prefix}\n"
        
        return result
    
    def get_system_status(self) -> str:
        """
        Get system status information.
        
        Returns:
            Formatted system status
        """
        try:
            # Test Weaviate connection
            weaviate_status = self.search_service.test_connection()
            
            result = "System Status:\n\n"
            result += f"🔗 Weaviate Connection:\n"
            result += f"  • URL: {weaviate_status['url']}\n"
            result += f"  • Live: {'✅' if weaviate_status['is_live'] else '❌'}\n"
            result += f"  • Ready: {'✅' if weaviate_status['is_ready'] else '❌'}\n"
            
            if 'version' in weaviate_status:
                result += f"  • Version: {weaviate_status['version']}\n"
            
            # List available classes
            classes = self.search_service.list_classes()
            result += f"\n📊 Indexed Classes: {len(classes)}\n"
            for class_prefix in classes[:5]:  # Show first 5
                info = self.search_service.get_class_info(class_prefix)
                count = info['document_count'] if info['exists'] else 0
                result += f"  • {class_prefix}: {count} docs\n"
            
            if len(classes) > 5:
                result += f"  • ... and {len(classes) - 5} more classes\n"
            
            # Monitored directories
            result += f"\n👁️ Monitored Directories: {len(self.monitored_directories)}\n"
            for directory in list(self.monitored_directories.keys())[:3]:  # Show first 3
                result += f"  • {directory}\n"
            
            if len(self.monitored_directories) > 3:
                result += f"  • ... and {len(self.monitored_directories) - 3} more\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return f"Failed to get system status: {e}"
    
    def create_tools(self) -> List[FunctionTool]:
        """Create FunctionTool instances for the agent."""
        tools = [
            FunctionTool.from_defaults(
                fn=self.search_indexed_content,
                name="search_indexed_content",
                description="Search indexed filesystem content semantically. Use this to find information in documents."
            ),
            FunctionTool.from_defaults(
                fn=self.list_indexed_classes,
                name="list_indexed_classes", 
                description="List all available indexed classes in the system with document counts."
            ),
            FunctionTool.from_defaults(
                fn=self.get_class_details,
                name="get_class_details",
                description="Get detailed information about a specific indexed class."
            ),
            FunctionTool.from_defaults(
                fn=self.list_directory_contents,
                name="list_directory_contents",
                description="List contents of a directory on the filesystem."
            ),
            FunctionTool.from_defaults(
                fn=self.read_file_content,
                name="read_file_content",
                description="Read content from a text file on the filesystem."
            ),
            FunctionTool.from_defaults(
                fn=self.get_monitored_directories,
                name="get_monitored_directories",
                description="Get list of currently monitored directories."
            ),
            FunctionTool.from_defaults(
                fn=self.get_system_status,
                name="get_system_status",
                description="Get system status including Weaviate connection and indexed classes."
            )
        ]
        
        return tools