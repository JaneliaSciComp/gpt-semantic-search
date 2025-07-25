"""
Agent configuration for local LLM integration.

Provides configuration management for local LLM server settings,
model paths, and agent-specific parameters.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for local LLM agent."""
    
    # Server configuration
    server_host: str = "localhost"
    server_port: int = 8080
    server_timeout: float = 60.0
    
    # Model configuration
    model_path: Optional[str] = None
    model_name: str = "your-model.gguf"
    
    # LLM parameters
    temperature: float = 0.1
    max_tokens: int = 512
    context_window: int = 4096
    
    # Agent parameters
    system_prompt: str = "You are a helpful assistant that can search and analyze filesystem content."
    verbose: bool = True
    enable_tools: bool = True
    
    # Fallback configuration
    fallback_to_openai: bool = True
    prefer_local_llm: bool = True
    
    @property
    def server_url(self) -> str:
        """Get the full server URL."""
        return f"http://{self.server_host}:{self.server_port}"
    
    @property
    def resolved_model_path(self) -> Optional[str]:
        """Get resolved model path."""
        if not self.model_path:
            return None
        
        path = Path(self.model_path).expanduser().resolve()
        return str(path) if path.exists() else None
    
    def get_llama_server_command(self) -> str:
        """Generate the llama-server command."""
        model_path = self.resolved_model_path or self.model_name
        
        return (
            f"llama-server "
            f"--model {model_path} "
            f"--host {self.server_host} "
            f"--port {self.server_port} "
            f"--ctx-size {self.context_window} "
            f"--chat-template chatml-function-calling"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Check if model file exists if path is specified
            if self.model_path:
                model_path = Path(self.model_path).expanduser().resolve()
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    return False
            
            # Validate port range
            if not (1024 <= self.server_port <= 65535):
                logger.error(f"Invalid port number: {self.server_port}")
                return False
            
            # Validate temperature
            if not (0.0 <= self.temperature <= 2.0):
                logger.error(f"Invalid temperature: {self.temperature}")
                return False
            
            # Validate max_tokens
            if self.max_tokens <= 0:
                logger.error(f"Invalid max_tokens: {self.max_tokens}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def get_default_config() -> AgentConfig:
    """Get default agent configuration with environment overrides."""
    config = AgentConfig()
    
    # Environment variable overrides
    if model_path := os.getenv('LLAMA_MODEL_PATH'):
        config.model_path = model_path
    
    if model_name := os.getenv('LLAMA_MODEL_NAME'):
        config.model_name = model_name
    
    if host := os.getenv('LLAMA_SERVER_HOST'):
        config.server_host = host
    
    if port := os.getenv('LLAMA_SERVER_PORT'):
        try:
            config.server_port = int(port)
        except ValueError:
            logger.warning(f"Invalid port in LLAMA_SERVER_PORT: {port}")
    
    if temperature := os.getenv('LLAMA_TEMPERATURE'):
        try:
            config.temperature = float(temperature)
        except ValueError:
            logger.warning(f"Invalid temperature in LLAMA_TEMPERATURE: {temperature}")
    
    if max_tokens := os.getenv('LLAMA_MAX_TOKENS'):
        try:
            config.max_tokens = int(max_tokens)
        except ValueError:
            logger.warning(f"Invalid max_tokens in LLAMA_MAX_TOKENS: {max_tokens}")
    
    # Boolean environment variables
    if os.getenv('LLAMA_VERBOSE', '').lower() in ('false', '0', 'no'):
        config.verbose = False
    
    if os.getenv('LLAMA_FALLBACK_OPENAI', '').lower() in ('false', '0', 'no'):
        config.fallback_to_openai = False
    
    if os.getenv('LLAMA_PREFER_LOCAL', '').lower() in ('false', '0', 'no'):
        config.prefer_local_llm = False
    
    return config


def find_model_files(search_paths: Optional[list] = None) -> list:
    """Find available GGUF model files."""
    if search_paths is None:
        search_paths = [
            Path.home() / "models",
            Path.cwd() / "models",
            Path("/usr/local/share/models"),
            Path("/opt/models"),
        ]
    
    models = []
    for search_path in search_paths:
        try:
            path = Path(search_path).expanduser().resolve()
            if path.exists() and path.is_dir():
                for model_file in path.rglob("*.gguf"):
                    models.append(str(model_file))
        except Exception as e:
            logger.debug(f"Error searching {search_path}: {e}")
    
    return sorted(models)


def suggest_model_path() -> Optional[str]:
    """Suggest a model path if available."""
    models = find_model_files()
    if models:
        return models[0]
    return None