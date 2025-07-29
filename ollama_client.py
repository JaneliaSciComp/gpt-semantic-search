"""Simple Ollama client for embeddings and LLM completions."""

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class SimpleOllamaAPI:
    """Simple Ollama API wrapper for embeddings and completions."""
    
    def __init__(self, 
                 embedding_model: str = "bge-m3:567m",
                 llm_model: str = "qwen3:4b",
                 base_url: str = "http://localhost:11434"):
        """Initialize Ollama client with models."""
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.base_url = base_url
        
        # Initialize embedding model
        self.ollama_embedding = OllamaEmbedding(
            model_name=embedding_model,
            base_url=base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )
        
        # Initialize LLM
        self.llm = Ollama(model=llm_model, request_timeout=60.0, base_url=base_url)
    
    def get_embedding(self, text: str) -> list:
        """Get embedding for a single text."""
        return self.ollama_embedding.get_text_embedding(text)
    
    def get_text_embedding_batch(self, texts: list, show_progress: bool = True) -> list:
        """Get embeddings for multiple texts."""
        return self.ollama_embedding.get_text_embedding_batch(texts, show_progress=show_progress)
    
    def get_query_embedding(self, query: str) -> list:
        """Get embedding for a query."""
        return self.ollama_embedding.get_query_embedding(query)
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text completion."""
        response = self.llm.complete(prompt)
        response_text = str(response)
        
        # Clean up thinking tags and verbose output from qwen3:4b
        import re
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'Thinking\.\.\..*?\.\.\.done thinking\.', '', response_text, flags=re.DOTALL)
        response_text = response_text.strip()
        
        return response_text
    
    def complete(self, prompt: str) -> str:
        """Complete text (alias for generate)."""
        return self.generate(prompt)


class LLM:
    """Legacy compatibility wrapper."""
    
    def __init__(self):
        self.ollama = SimpleOllamaAPI()
    
    def get_models(self):
        """Get available Ollama models."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                result = response.json()
                models = [model.get("name", "") for model in result.get("models", []) if model.get("name")]
            else:
                models = []
        except:
            models = []
        
        if not models:
            # Fallback to default models if Ollama is not available
            models = ["qwen:4b", "bge-m3:567m"]
        
        return sorted(models)