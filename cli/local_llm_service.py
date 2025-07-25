"""
Local LLM service using llama-cpp-python.

Provides LlamaCPP LLM initialization and basic FunctionAgent setup
for agentic workflow integration with the filesystem RAG application.
"""

import logging
import time
import httpx
from typing import Optional, List, Dict, Any
from pathlib import Path

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool

from .agent_config import AgentConfig, get_default_config

logger = logging.getLogger(__name__)

# Complete Python code snippet for local LLM setup as requested
LLAMA_SERVER_SETUP_CODE = '''
"""
Complete Python code snippet for local LLM with tool-calling support and LlamaIndex FunctionAgent.

Requirements:
- llama-cpp-python installed and llama-server available in PATH
- your-model.gguf file in current directory or specify full path
- Port 8080 available
"""

# Required imports
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool

# Step 1: llama-server command (run this in terminal before executing Python code)
# IMPORTANT: Start this command in terminal first:
LLAMA_SERVER_COMMAND = "llama-server --model your-model.gguf --host localhost --port 8080 --ctx-size 4096 --chat-template chatml-function-calling"

# Step 2: LlamaCPP LLM initialization
llm = LlamaCPP(
    api_base_url="http://localhost:8080",
    temperature=0.1,
    max_tokens=512,
    request_timeout=60.0,
    # Additional parameters for better performance
    context_window=4096,
    # Ensure tool calling support
    is_chat_model=True,
)

# Step 3: Basic FunctionAgent initialization
agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[],  # Empty list initially - tools will be added in subsequent steps
    llm=llm,
    system_prompt="You are a helpful assistant.",
    verbose=True,
)

# Create the agent runner
agent = AgentRunner(agent_worker)

# Example usage:
# response = agent.query("Hello, how are you?")
# print(response.response)
'''


class LocalLLMService:
    """Service for managing local LLM connections and agent initialization."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_default_config()
        self.llm: Optional[LlamaCPP] = None
        self.agent: Optional[AgentRunner] = None
        self._connection_validated = False
    
    def get_server_command(self) -> str:
        """Get the llama-server startup command."""
        return self.config.get_llama_server_command()
    
    def validate_server_connection(self) -> bool:
        """Validate connection to llama-server."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.config.server_url}/health")
                if response.status_code == 200:
                    self._connection_validated = True
                    logger.info(f"Connected to llama-server at {self.config.server_url}")
                    return True
                else:
                    logger.error(f"Server returned status {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to llama-server: {e}")
            return False
    
    def wait_for_server(self, max_wait: int = 30) -> bool:
        """Wait for server to become available."""
        logger.info(f"Waiting for llama-server at {self.config.server_url}...")
        
        for i in range(max_wait):
            if self.validate_server_connection():
                return True
            time.sleep(1)
        
        logger.error(f"Server not available after {max_wait} seconds")
        return False
    
    def initialize_llm(self) -> Optional[LlamaCPP]:
        """Initialize LlamaCPP LLM with connection validation."""
        if not self._connection_validated and not self.validate_server_connection():
            logger.error("Cannot initialize LLM: server connection failed")
            return None
        
        try:
            self.llm = LlamaCPP(
                api_base_url=self.config.server_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                request_timeout=self.config.server_timeout,
                context_window=self.config.context_window,
                is_chat_model=True,
            )
            
            logger.info("LlamaCPP LLM initialized successfully")
            return self.llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCPP LLM: {e}")
            return None
    
    def initialize_agent(self, tools: Optional[List[FunctionTool]] = None) -> Optional[AgentRunner]:
        """Initialize FunctionAgent with tools."""
        if not self.llm:
            self.llm = self.initialize_llm()
            if not self.llm:
                return None
        
        try:
            # Create agent worker with tools
            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=tools or [],
                llm=self.llm,
                system_prompt=self.config.system_prompt,
                verbose=self.config.verbose,
            )
            
            # Create agent runner
            self.agent = AgentRunner(agent_worker)
            
            logger.info("FunctionAgent initialized successfully")
            return self.agent
            
        except Exception as e:
            logger.error(f"Failed to initialize FunctionAgent: {e}")
            return None
    
    def test_llm_connection(self) -> Dict[str, Any]:
        """Test LLM connection and return status."""
        status = {
            'server_url': self.config.server_url,
            'connected': False,
            'llm_initialized': False,
            'agent_initialized': False,
        }
        
        # Test server connection
        if self.validate_server_connection():
            status['connected'] = True
            
            # Test LLM initialization
            if self.initialize_llm():
                status['llm_initialized'] = True
                
                # Test a simple query
                try:
                    response = self.llm.complete("Hello")
                    status['test_response'] = str(response)
                    logger.info("LLM test query successful")
                except Exception as e:
                    status['test_error'] = str(e)
                    logger.error(f"LLM test query failed: {e}")
        
        return status
    
    def get_setup_code(self) -> str:
        """Get the complete setup code snippet."""
        return LLAMA_SERVER_SETUP_CODE
    
    def is_available(self) -> bool:
        """Check if local LLM is available and working."""
        return (
            self._connection_validated or self.validate_server_connection()
        ) and self.llm is not None
    
    def shutdown(self):
        """Clean up resources."""
        if self.agent:
            # Agent cleanup if needed
            self.agent = None
            
        if self.llm:
            # LLM cleanup if needed  
            self.llm = None
            
        self._connection_validated = False
        logger.info("Local LLM service shut down")


def create_basic_agent(
    model_path: str = "your-model.gguf",
    server_port: int = 8080,
    temperature: float = 0.1,
    max_tokens: int = 512,
    system_prompt: str = "You are a helpful assistant.",
    verbose: bool = True
) -> Optional[AgentRunner]:
    """
    Create a basic FunctionAgent with local LLM.
    
    This is a simplified version that matches the original prompt requirements.
    """
    # Create configuration
    config = AgentConfig(
        model_name=model_path,
        server_port=server_port,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        verbose=verbose
    )
    
    # Create service and initialize agent
    service = LocalLLMService(config)
    
    # Print server command for user
    print("Start llama-server with this command:")
    print(service.get_server_command())
    print()
    
    # Wait for server
    if not service.wait_for_server():
        logger.error("Server not available")
        return None
    
    # Initialize agent
    return service.initialize_agent()


# Example usage following the original prompt requirements
if __name__ == "__main__":
    print("=== Local LLM Setup Code ===")
    print(LLAMA_SERVER_SETUP_CODE)
    print()
    
    print("=== Interactive Example ===")
    
    # Create the basic agent as specified in requirements
    agent = create_basic_agent(
        model_path="your-model.gguf",
        server_port=8080,
        temperature=0.1,
        max_tokens=512,
        system_prompt="You are a helpful assistant.",
        verbose=True
    )
    
    if agent:
        print("Agent initialized successfully!")
        
        # Test the agent
        try:
            response = agent.query("Hello, how are you?")
            print(f"Agent response: {response.response}")
        except Exception as e:
            print(f"Agent query failed: {e}")
    else:
        print("Agent initialization failed")