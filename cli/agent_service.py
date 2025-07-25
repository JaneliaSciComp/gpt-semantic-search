"""
Agent service for filesystem RAG operations.

Orchestrates the local LLM agent with filesystem tools to provide
an agentic interface for exploring and searching indexed content.
"""

import logging
from typing import Optional, Dict, Any, List

from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI

from .local_llm_service import LocalLLMService
from .agent_tools import FilesystemAgentTools
from .agent_config import AgentConfig, get_default_config
from .search_service import SearchService

logger = logging.getLogger(__name__)


class AgentService:
    """
    Orchestrates the agentic workflow for filesystem RAG.
    
    Manages local LLM agent initialization, tool integration,
    and provides fallback to OpenAI when local LLM is unavailable.
    """
    
    def __init__(
        self,
        search_service: SearchService,
        config: Optional[AgentConfig] = None,
        monitored_directories: Optional[Dict[str, str]] = None
    ):
        self.search_service = search_service
        self.config = config or get_default_config()
        self.monitored_directories = monitored_directories or {}
        
        # Services
        self.local_llm_service: Optional[LocalLLMService] = None
        self.agent_tools: Optional[FilesystemAgentTools] = None
        self.agent: Optional[AgentRunner] = None
        
        # State
        self.is_initialized = False
        self.using_local_llm = False
        self.initialization_error: Optional[str] = None
    
    def initialize(self) -> bool:
        """
        Initialize the agent service.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize agent tools
            self.agent_tools = FilesystemAgentTools(
                search_service=self.search_service,
                monitored_directories=self.monitored_directories
            )
            
            # Try to initialize local LLM first
            if self.config.prefer_local_llm:
                success = self._initialize_local_agent()
                if success:
                    self.using_local_llm = True
                    self.is_initialized = True
                    logger.info("Agent service initialized with local LLM")
                    return True
                elif not self.config.fallback_to_openai:
                    self.initialization_error = "Local LLM failed and fallback disabled"
                    return False
            
            # Fallback to OpenAI agent
            if self.config.fallback_to_openai:
                success = self._initialize_openai_agent()
                if success:
                    self.using_local_llm = False
                    self.is_initialized = True
                    logger.info("Agent service initialized with OpenAI fallback")
                    return True
            
            self.initialization_error = "Both local LLM and OpenAI fallback failed"
            return False
            
        except Exception as e:
            self.initialization_error = f"Agent initialization failed: {e}"
            logger.error(self.initialization_error)
            return False
    
    def _initialize_local_agent(self) -> bool:
        """Initialize agent with local LLM."""
        try:
            self.local_llm_service = LocalLLMService(self.config)
            
            # Wait for server
            if not self.local_llm_service.wait_for_server(max_wait=10):
                logger.warning("Local LLM server not available")
                return False
            
            # Create tools
            tools = self.agent_tools.create_tools()
            
            # Initialize agent
            self.agent = self.local_llm_service.initialize_agent(tools)
            
            return self.agent is not None
            
        except Exception as e:
            logger.error(f"Local agent initialization failed: {e}")
            return False
    
    def _initialize_openai_agent(self) -> bool:
        """Initialize agent with OpenAI as fallback."""
        try:
            # Import OpenAI agent components
            from llama_index.core.agent import FunctionCallingAgentWorker
            from llama_index.core.agent import AgentRunner
            
            # Create OpenAI LLM
            llm = OpenAI(
                model="gpt-4o",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Create tools
            tools = self.agent_tools.create_tools()
            
            # Create agent worker
            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=tools,
                llm=llm,
                system_prompt=self.config.system_prompt,
                verbose=self.config.verbose,
            )
            
            # Create agent runner
            self.agent = AgentRunner(agent_worker)
            
            return True
            
        except Exception as e:
            logger.error(f"OpenAI agent initialization failed: {e}")
            return False
    
    def query(self, message: str) -> str:
        """
        Query the agent with a message.
        
        Args:
            message: User message/query
            
        Returns:
            Agent response
        """
        if not self.is_initialized:
            if not self.initialize():
                return f"Agent not available: {self.initialization_error}"
        
        try:
            response = self.agent.query(message)
            return str(response.response)
            
        except Exception as e:
            error_msg = f"Agent query failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def chat(self, message: str) -> str:
        """
        Chat with the agent (alias for query).
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        return self.query(message)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent service status.
        
        Returns:
            Status information dictionary
        """
        status = {
            'initialized': self.is_initialized,
            'using_local_llm': self.using_local_llm,
            'config': self.config.to_dict(),
        }
        
        if self.initialization_error:
            status['error'] = self.initialization_error
        
        if self.local_llm_service:
            status['local_llm_status'] = self.local_llm_service.test_llm_connection()
        
        if self.agent_tools:
            status['tools_available'] = len(self.agent_tools.create_tools())
            status['monitored_directories'] = len(self.monitored_directories)
        
        return status
    
    def get_server_command(self) -> Optional[str]:
        """
        Get the llama-server startup command.
        
        Returns:
            Server command string or None if not using local LLM
        """
        if self.config.prefer_local_llm:
            return self.config.get_llama_server_command()
        return None
    
    def restart(self) -> bool:
        """
        Restart the agent service.
        
        Returns:
            True if restart successful
        """
        self.shutdown()
        return self.initialize()
    
    def shutdown(self):
        """Shutdown the agent service."""
        if self.local_llm_service:
            self.local_llm_service.shutdown()
            
        self.agent = None
        self.agent_tools = None
        self.is_initialized = False
        self.using_local_llm = False
        self.initialization_error = None
        
        logger.info("Agent service shut down")
    
    def update_monitored_directories(self, directories: Dict[str, str]):
        """
        Update monitored directories.
        
        Args:
            directories: Dictionary mapping directory paths to class prefixes
        """
        self.monitored_directories = directories.copy()
        
        # Update tools if initialized
        if self.agent_tools:
            self.agent_tools.monitored_directories = self.monitored_directories
    
    def is_available(self) -> bool:
        """Check if agent service is available."""
        return self.is_initialized and self.agent is not None
    
    def get_example_queries(self) -> List[str]:
        """Get example queries for the agent."""
        return [
            "What indexed content is available?",
            "Search for 'configuration' in the indexed documents",
            "List all monitored directories",
            "Show me the system status",
            "What files are in the current directory?",
            "Find documents about 'API usage'",
            "Get details about the Filesystem class",
        ]
    
    def get_help_text(self) -> str:
        """Get help text for using the agent."""
        llm_type = "Local LLM" if self.using_local_llm else "OpenAI"
        
        help_text = f"""
🤖 Filesystem RAG Agent ({llm_type})

The agent can help you explore and search your indexed filesystem content.
It has access to the following capabilities:

📊 Content Search:
• Search indexed documents semantically
• List available indexed classes  
• Get details about specific classes

📁 Filesystem Access:
• List directory contents
• Read file contents
• Show monitored directories

🔧 System Information:
• Get system status
• Check Weaviate connection
• Show indexing statistics

Example queries:
"""
        
        for query in self.get_example_queries():
            help_text += f"• {query}\n"
        
        if self.using_local_llm and self.local_llm_service:
            help_text += f"\n🖥️ Local LLM Server:\n"
            help_text += f"Command: {self.get_server_command()}\n"
            help_text += f"URL: {self.config.server_url}\n"
        
        return help_text


def create_agent_service(
    search_service: SearchService,
    monitored_directories: Optional[Dict[str, str]] = None,
    **config_kwargs
) -> AgentService:
    """
    Create and initialize an agent service.
    
    Args:
        search_service: Search service instance
        monitored_directories: Dictionary of monitored directories
        **config_kwargs: Configuration overrides
        
    Returns:
        Initialized agent service
    """
    config = get_default_config()
    
    # Apply config overrides
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    service = AgentService(
        search_service=search_service,
        config=config,
        monitored_directories=monitored_directories
    )
    
    service.initialize()
    return service