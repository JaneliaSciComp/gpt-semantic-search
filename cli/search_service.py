"""
Search service for filesystem RAG CLI.

Provides semantic search capabilities using Weaviate vector database
and OpenAI models, following the proven patterns from the web service.
"""

import logging
import re
import textwrap
import time
from typing import Optional, List, Dict, Any

import weaviate
from ollama_client import SimpleOllamaAPI

logger = logging.getLogger(__name__)

# Model configuration (following web service patterns)
EMBED_MODEL_NAME = "bge-m3:567m"
CONTEXT_WINDOW = 128000
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1


class SearchService:
    """
    Semantic search service for filesystem RAG.
    
    Provides search capabilities across indexed filesystem content using
    Weaviate vector database and OpenAI models.
    """
    
    def __init__(self, weaviate_url: str, debug: bool = False):
        self.weaviate_url = weaviate_url
        self.debug = debug
        self.weaviate_client: Optional[weaviate.Client] = None
        self.ollama = SimpleOllamaAPI()
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
    
    def _get_weaviate_client(self) -> weaviate.Client:
        """Get Weaviate client with connection validation."""
        if self.weaviate_client is None:
            self.weaviate_client = weaviate.Client(self.weaviate_url)
            
            if not self.weaviate_client.is_live():
                raise ConnectionError(f"Weaviate is not live at {self.weaviate_url}")
        
        return self.weaviate_client
    
    def _search_weaviate(
        self,
        query: str,
        class_prefix: str,
        num_results: int = 3
    ) -> List[Dict]:
        """Direct Weaviate search without LlamaIndex."""
        client = self._get_weaviate_client()
        class_name = f"{class_prefix}_Node"
        
        # Get query embedding
        query_embedding = self.ollama.get_embedding(query)
        
        # Perform vector search
        result = (
            client.query
            .get(class_name, ["text", "title", "link", "source", "ref_doc_id"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(num_results)
            .do()
        )
        
        if not result.get("data", {}).get("Get", {}).get(class_name):
            return []
        
        return result["data"]["Get"][class_name]
    
    def _escape_text(self, text: str) -> str:
        """Escape special characters for display."""
        text = re.sub("<", "&lt;", text)
        text = re.sub(">", "&gt;", text)
        text = re.sub("([_#])", r"\\\\\\1", text)
        return text
    
    
    def search(
        self,
        query: str,
        class_prefix: str,
        temperature: float = 0.1,
        search_alpha: float = 0.8,
        num_results: int = 3
    ) -> str:
        """
        Search indexed documents and return formatted response.
        
        Args:
            query: Search query
            class_prefix: Weaviate class prefix to search
            temperature: LLM temperature for response generation
            search_alpha: Hybrid search balance (ignored for now)
            num_results: Number of results to retrieve
            
        Returns:
            Formatted response with sources
            
        Raises:
            ConnectionError: If Weaviate is not accessible
            RuntimeError: If search fails
        """
        start_time = time.time()
        
        try:
            # Clean query
            clean_query = re.sub('"', "", query)
            
            # Get search results from Weaviate
            search_results = self._search_weaviate(clean_query, class_prefix, num_results)
            
            if not search_results:
                return f"No results found for: '{query}'"
            
            # Build context for LLM
            context = "Context:\n"
            for i, result in enumerate(search_results, 1):
                text = result.get("text", "")[:500]  # Limit context size
                title = result.get("title", "Untitled")
                context += f"{i}. {title}: {text}\n\n"
            
            # Generate response with Ollama
            prompt = f"""Based on the following context, answer the user's question: "{clean_query}"

{context}

Provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
            
            response_text = self.ollama.generate(prompt, temperature=temperature)
            
            # Format response with sources
            formatted_response = f"{response_text}\n\nSources:\n\n"
            
            for result in search_results:
                text = result.get("text", "")
                text = re.sub(r"\n+", " ", text)
                text = textwrap.shorten(text, width=100, placeholder="...")
                text = self._escape_text(text)
                
                source = result.get("source", "Unknown")
                title = result.get("title", "Untitled")
                link = result.get("link", "")
                
                if link:
                    formatted_response += f"* {source}: {title}\n  File: {link}\n  {text}\n\n---\n\n"
                else:
                    formatted_response += f"* {source}: {title}\n  {text}\n\n---\n\n"
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"Search failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def search_multiple_classes(
        self,
        query: str,
        class_prefixes: List[str],
        temperature: float = 0.1,
        search_alpha: float = 0.8,
        num_results: int = 3
    ) -> str:
        """
        Search across multiple classes and combine results.
        
        Args:
            query: Search query
            class_prefixes: List of class prefixes to search
            temperature: LLM temperature
            search_alpha: Hybrid search balance
            num_results: Number of results per class
            
        Returns:
            Combined formatted response
        """
        if not class_prefixes:
            return "No classes configured for searching."
        
        results = []
        
        for class_prefix in class_prefixes:
            try:
                result = self.search(query, class_prefix, temperature, search_alpha, num_results)
                
                # Skip empty results
                if result and "No results found" not in result:
                    results.append(f"=== Results from {class_prefix} ===\n{result}")
                    
            except Exception as e:
                logger.error(f"Search failed for class {class_prefix}: {e}")
                continue
        
        if results:
            return "\n\n".join(results)
        else:
            return f"No results found for: '{query}'"
    
    def check_class_exists(self, class_prefix: str) -> bool:
        """
        Check if a Weaviate class exists.
        
        Args:
            class_prefix: Class prefix to check
            
        Returns:
            True if class exists, False otherwise
        """
        try:
            client = self._get_weaviate_client()
            schema = client.schema.get()
            
            # Check if any class starts with the prefix
            class_name = f"{class_prefix}_Node"
            for class_def in schema.get('classes', []):
                if class_def['class'] == class_name:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check class existence: {e}")
            return False
    
    def list_classes(self) -> List[str]:
        """
        List all available Weaviate classes.
        
        Returns:
            List of class names
        """
        try:
            client = self._get_weaviate_client()
            schema = client.schema.get()
            
            classes = []
            for class_def in schema.get('classes', []):
                class_name = class_def['class']
                # Extract prefix from class name (remove _Node suffix)
                if class_name.endswith('_Node'):
                    prefix = class_name[:-5]  # Remove '_Node'
                    classes.append(prefix)
                else:
                    classes.append(class_name)
            
            return sorted(classes)
            
        except Exception as e:
            logger.error(f"Failed to list classes: {e}")
            return []
    
    def get_class_info(self, class_prefix: str) -> Dict[str, Any]:
        """
        Get information about a specific class.
        
        Args:
            class_prefix: Class prefix to get info for
            
        Returns:
            Dictionary with class information
        """
        try:
            client = self._get_weaviate_client()
            class_name = f"{class_prefix}_Node"
            
            # Get class schema
            try:
                schema = client.schema.get_class_schema(class_name)
            except AttributeError:
                # Fallback for older Weaviate versions
                all_schema = client.schema.get()
                schema = None
                for class_def in all_schema.get('classes', []):
                    if class_def['class'] == class_name:
                        schema = class_def
                        break
                if schema is None:
                    schema = {'properties': []}
            
            # Count objects in class
            result = client.query.aggregate(class_name).with_meta_count().do()
            count = 0
            if result.get('data', {}).get('Aggregate', {}).get(class_name):
                meta = result['data']['Aggregate'][class_name][0].get('meta', {})
                count = meta.get('count', 0)
            
            return {
                'class_name': class_name,
                'class_prefix': class_prefix,
                'document_count': count,
                'properties': [prop['name'] for prop in schema.get('properties', [])],
                'exists': True
            }
            
        except Exception as e:
            logger.error(f"Failed to get class info for {class_prefix}: {e}")
            return {
                'class_name': f"{class_prefix}_Node",
                'class_prefix': class_prefix,
                'document_count': 0,
                'properties': [],
                'exists': False,
                'error': str(e)
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Weaviate and return status information.
        
        Returns:
            Dictionary with connection status
        """
        try:
            client = self._get_weaviate_client()
            
            # Test basic connectivity
            is_live = client.is_live()
            is_ready = client.is_ready()
            
            status = {
                'url': self.weaviate_url,
                'is_live': is_live,
                'is_ready': is_ready,
                'connected': True
            }
            
            if is_live:
                try:
                    meta = client.get_meta()
                    status['version'] = meta.get('version', 'unknown')
                    status['modules'] = meta.get('modules', {})
                except:
                    pass
            
            return status
            
        except Exception as e:
            return {
                'url': self.weaviate_url,
                'is_live': False,
                'is_ready': False,
                'connected': False,
                'error': str(e)
            }
    
    def __str__(self) -> str:
        return f"SearchService(weaviate_url='{self.weaviate_url}')"