"""
Configuration management for interactive filesystem RAG CLI.

Handles persistent storage and loading of directory monitoring configuration,
settings, and user preferences.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".filesystem_rag_config.json"
DEFAULT_WEAVIATE_URL = "http://localhost:8777"


class DirectoryConfig:
    """Configuration for a single monitored directory."""
    
    def __init__(
        self,
        path: str,
        class_prefix: str,
        enabled: bool = True,
        last_indexed: Optional[str] = None
    ):
        self.path = str(Path(path).absolute())
        self.class_prefix = class_prefix
        self.enabled = enabled
        self.last_indexed = last_indexed or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "class_prefix": self.class_prefix,
            "enabled": self.enabled,
            "last_indexed": self.last_indexed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectoryConfig":
        """Create from dictionary loaded from JSON."""
        return cls(
            path=data["path"],
            class_prefix=data["class_prefix"],
            enabled=data.get("enabled", True),
            last_indexed=data.get("last_indexed")
        )
    
    def __str__(self) -> str:
        status = "✓ Active" if self.enabled else "✗ Disabled"
        return f"{self.path} ({self.class_prefix}) - {status}"


class RAGConfig:
    """Main configuration class for filesystem RAG CLI."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.weaviate_url = DEFAULT_WEAVIATE_URL
        self.directories: List[DirectoryConfig] = []
        self.settings = {
            "debounce_ms": 1000,
            "debug": False,
            "auto_index": True,
            "search_alpha": 0.8,
            "temperature": 0.1,
            "num_results": 10
        }
        
        # Load existing config if it exists
        if self.config_path.exists():
            self.load()
    
    def add_directory(
        self,
        path: str,
        class_prefix: str,
        enabled: bool = True
    ) -> DirectoryConfig:
        """
        Add a new directory to monitor.
        
        Args:
            path: Directory path to monitor
            class_prefix: Weaviate class prefix for this directory
            enabled: Whether monitoring is enabled
            
        Returns:
            DirectoryConfig object
            
        Raises:
            ValueError: If directory already exists or path is invalid
        """
        abs_path = str(Path(path).absolute())
        
        # Check if directory already exists
        for dir_config in self.directories:
            if dir_config.path == abs_path:
                raise ValueError(f"Directory already monitored: {abs_path}")
            if dir_config.class_prefix == class_prefix:
                raise ValueError(f"Class prefix already in use: {class_prefix}")
        
        # Validate path exists
        if not Path(abs_path).exists():
            raise ValueError(f"Directory does not exist: {abs_path}")
        
        if not Path(abs_path).is_dir():
            raise ValueError(f"Path is not a directory: {abs_path}")
        
        dir_config = DirectoryConfig(abs_path, class_prefix, enabled)
        self.directories.append(dir_config)
        
        logger.info(f"Added directory: {dir_config}")
        return dir_config
    
    def remove_directory(self, path: str) -> bool:
        """
        Remove a directory from monitoring.
        
        Args:
            path: Directory path to remove
            
        Returns:
            True if removed, False if not found
        """
        abs_path = str(Path(path).absolute())
        
        for i, dir_config in enumerate(self.directories):
            if dir_config.path == abs_path:
                removed = self.directories.pop(i)
                logger.info(f"Removed directory: {removed}")
                return True
        
        return False
    
    def get_directory(self, path: str) -> Optional[DirectoryConfig]:
        """Get directory config by path."""
        abs_path = str(Path(path).absolute())
        
        for dir_config in self.directories:
            if dir_config.path == abs_path:
                return dir_config
        
        return None
    
    def get_directory_by_prefix(self, class_prefix: str) -> Optional[DirectoryConfig]:
        """Get directory config by class prefix."""
        for dir_config in self.directories:
            if dir_config.class_prefix == class_prefix:
                return dir_config
        
        return None
    
    def list_directories(self, enabled_only: bool = False) -> List[DirectoryConfig]:
        """
        Get list of monitored directories.
        
        Args:
            enabled_only: If True, only return enabled directories
            
        Returns:
            List of DirectoryConfig objects
        """
        if enabled_only:
            return [d for d in self.directories if d.enabled]
        return self.directories.copy()
    
    def update_last_indexed(self, path: str) -> None:
        """Update the last indexed time for a directory."""
        dir_config = self.get_directory(path)
        if dir_config:
            dir_config.last_indexed = datetime.now().isoformat()
    
    def set_setting(self, key: str, value: Any) -> None:
        """Update a setting value."""
        if key in self.settings:
            self.settings[key] = value
            logger.debug(f"Updated setting {key} = {value}")
        else:
            logger.warning(f"Unknown setting: {key}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self.settings.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization."""
        return {
            "weaviate_url": self.weaviate_url,
            "directories": [d.to_dict() for d in self.directories],
            "settings": self.settings.copy()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        self.weaviate_url = data.get("weaviate_url", DEFAULT_WEAVIATE_URL)
        
        # Load directories
        self.directories = []
        for dir_data in data.get("directories", []):
            try:
                dir_config = DirectoryConfig.from_dict(dir_data)
                # Validate directory still exists
                if Path(dir_config.path).exists():
                    self.directories.append(dir_config)
                else:
                    logger.warning(f"Skipping non-existent directory: {dir_config.path}")
            except Exception as e:
                logger.error(f"Failed to load directory config: {e}")
        
        # Load settings
        if "settings" in data:
            self.settings.update(data["settings"])
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(self.config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load(self) -> None:
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                logger.info(f"No configuration file found at: {self.config_path}")
                return
            
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.from_dict(data)
            logger.info(f"Configuration loaded from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Don't raise - continue with default config
    
    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create a backup of the current configuration.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_path.with_name(f"{self.config_path.stem}_backup_{timestamp}.json")
        
        with open(backup_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration backed up to: {backup_path}")
        return backup_path
    
    def validate(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check Weaviate URL format
        if not self.weaviate_url.startswith(("http://", "https://")):
            errors.append(f"Invalid Weaviate URL format: {self.weaviate_url}")
        
        # Check directories
        class_prefixes = set()
        for dir_config in self.directories:
            # Check path exists
            if not Path(dir_config.path).exists():
                errors.append(f"Directory does not exist: {dir_config.path}")
            elif not Path(dir_config.path).is_dir():
                errors.append(f"Path is not a directory: {dir_config.path}")
            
            # Check for duplicate class prefixes
            if dir_config.class_prefix in class_prefixes:
                errors.append(f"Duplicate class prefix: {dir_config.class_prefix}")
            class_prefixes.add(dir_config.class_prefix)
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = [
            f"Filesystem RAG Configuration:",
            f"  Weaviate URL: {self.weaviate_url}",
            f"  Config Path: {self.config_path}",
            f"  Directories: {len(self.directories)}"
        ]
        
        for dir_config in self.directories:
            lines.append(f"    - {dir_config}")
        
        return "\n".join(lines)