"""
Modern RAG Integration Layer

Provides easy integration of all modern RAG components with the existing
JaneliaGPT system. This module serves as the main entry point for using
the enhanced RAG capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Import our modern components
from chunking.advanced_chunkers import (
    HierarchicalChunker, SyntaxAwareChunker, AdaptiveChunker,
    ContentType, DocumentChunk
)
from chunking.hyde_query_transformer import HyDEIntegration
from indexing.modern_weaviate_indexer import (
    ModernWeaviateIndexer, ModernIndexingPipeline,
    EmbeddingType, EmbeddingConfig
)
from agents.query_router import AgenticQueryProcessor
from agents.fast_query_processor import FastQueryProcessor, ProcessingMode, StreamingQueryEngine
from retrieval.enhanced_retrieval import (
    EnhancedRetrievalPipeline, ModernRAGRetriever,
    RetrievalPipelineConfig
)

logger = logging.getLogger(__name__)

@dataclass
class ModernRAGConfig:
    """Configuration for the modern RAG system."""
    # Weaviate configuration
    weaviate_url: str = "http://localhost:8777"
    class_name: str = "Janelia_Node"
    
    # Model configurations
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.3
    
    # Chunking configuration
    chunking_strategy: str = "hierarchical"  # hierarchical, syntax_aware, adaptive
    max_chunk_size: int = 512
    min_chunk_size: int = 100
    overlap_size: int = 50
    
    # Multi-vector configuration
    enable_multi_vector: bool = True
    enable_code_embeddings: bool = True
    enable_conversational_embeddings: bool = True
    
    # Query enhancement
    enable_hyde: bool = True
    enable_query_routing: bool = True
    
    # Retrieval configuration
    enable_enhanced_retrieval: bool = True
    dense_top_k: int = 50
    sparse_top_k: int = 30
    final_top_k: int = 10
    
    # Cross-encoder reranking
    enable_reranking: bool = True
    rerank_threshold: float = 0.5
    
    # Performance optimization controls
    agentic_processing_mode: str = "balanced"  # fast, balanced, comprehensive
    enable_streaming: bool = True
    skip_complex_routing: bool = False
    hyde_timeout_seconds: float = 3.0
    routing_timeout_seconds: float = 2.0
    enable_parallel_processing: bool = True

class ModernRAGSystem:
    """
    Complete modern RAG system that can be integrated with existing JaneliaGPT.
    
    This class provides a high-level interface for:
    - Advanced document chunking and indexing
    - Multi-vector embeddings
    - HyDE query transformation
    - Agentic query routing
    - Enhanced multi-stage retrieval
    """
    
    def __init__(self, config: Optional[ModernRAGConfig] = None):
        """Initialize the modern RAG system."""
        self.config = config or ModernRAGConfig()
        
        # Initialize core components
        self.llm = OpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )
        self.embedding_model = OpenAIEmbedding(
            model=self.config.embedding_model
        )
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = self.config.max_chunk_size
        
        # Initialize advanced components
        self._initialize_components()
        
        logger.info("Modern RAG system initialized")
    
    def _initialize_components(self):
        """Initialize all modern RAG components."""
        # Initialize chunkers
        self.chunkers = {
            'hierarchical': HierarchicalChunker(
                max_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size,
                overlap_size=self.config.overlap_size
            ),
            'syntax_aware': SyntaxAwareChunker(
                target_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size
            ),
            'adaptive': AdaptiveChunker(
                base_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size
            )
        }
        
        # Configure embedding types
        embedding_configs = {}
        if self.config.enable_multi_vector:
            embedding_configs[EmbeddingType.DENSE] = EmbeddingConfig(
                name="dense",
                model=self.config.embedding_model,
                dimensions=3072,
                embedding_type=EmbeddingType.DENSE,
                description="Main content embedding"
            )
            embedding_configs[EmbeddingType.SPARSE] = EmbeddingConfig(
                name="sparse",
                model="text-embedding-3-small",
                dimensions=1536,
                embedding_type=EmbeddingType.SPARSE,
                description="Compact embedding for efficiency"
            )
            
            if self.config.enable_code_embeddings:
                embedding_configs[EmbeddingType.CODE] = EmbeddingConfig(
                    name="code",
                    model=self.config.embedding_model,
                    dimensions=3072,
                    embedding_type=EmbeddingType.CODE,
                    description="Code-specific embedding"
                )
            
            if self.config.enable_conversational_embeddings:
                embedding_configs[EmbeddingType.CONVERSATIONAL] = EmbeddingConfig(
                    name="conversational",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    embedding_type=EmbeddingType.CONVERSATIONAL,
                    description="Conversational embedding for Slack"
                )
        
        # Initialize indexer
        self.indexer = ModernWeaviateIndexer(
            weaviate_url=self.config.weaviate_url,
            class_name=self.config.class_name,
            embedding_configs=embedding_configs if self.config.enable_multi_vector else None
        )
        
        # Initialize indexing pipeline with safe chunk sizes for OpenAI embedding limits
        chunker_config = {
            "max_chunk_size": 6000,  # Safe margin below 8192 token limit
            "min_chunk_size": 100,
            "overlap_size": 200
        }
        
        self.indexing_pipeline = ModernIndexingPipeline(
            weaviate_url=self.config.weaviate_url,
            class_name=self.config.class_name,
            chunker_config=chunker_config,
            embedding_configs=embedding_configs if self.config.enable_multi_vector else None
        )
        
        # Initialize query enhancement components
        if self.config.enable_hyde:
            self.hyde_integration = HyDEIntegration(
                self.config.llm_model,
                self.config.embedding_model
            )
        
        if self.config.enable_query_routing:
            self.query_processor = AgenticQueryProcessor(
                self.config.llm_model
            )
            
            # Initialize fast processor for performance optimization
            processing_mode = ProcessingMode(self.config.agentic_processing_mode)
            self.fast_query_processor = FastQueryProcessor(
                llm_model=self.config.llm_model,
                processing_mode=processing_mode,
                enable_streaming=self.config.enable_streaming,
                enable_parallel=self.config.enable_parallel_processing
            )
        
        # Initialize enhanced retrieval pipeline
        if self.config.enable_enhanced_retrieval:
            retrieval_config = RetrievalPipelineConfig(
                dense_top_k=self.config.dense_top_k,
                sparse_top_k=self.config.sparse_top_k,
                final_top_k=self.config.final_top_k,
                enable_cross_encoder=self.config.enable_reranking,
                cross_encoder_threshold=self.config.rerank_threshold
            )
            
            self.retrieval_pipeline = EnhancedRetrievalPipeline(
                indexer=self.indexer,
                config=retrieval_config,
                llm_model=self.config.llm_model
            )
            
            self.modern_retriever = ModernRAGRetriever(
                pipeline=self.retrieval_pipeline
            )
    
    def create_schema(self, delete_existing: bool = False) -> None:
        """Create or update the Weaviate schema."""
        self.indexer.create_schema(delete_existing=delete_existing)
    
    def index_documents(self, 
                       documents: List[Document],
                       content_type: ContentType = ContentType.PLAIN_TEXT,
                       use_advanced_chunking: bool = True) -> None:
        """
        Index documents using modern chunking and multi-vector embeddings.
        
        Args:
            documents: List of documents to index
            content_type: Type of content for optimized processing
            use_advanced_chunking: Whether to use advanced chunking strategies
        """
        try:
            if use_advanced_chunking:
                # Use advanced chunking pipeline
                self.indexing_pipeline.index_documents_with_advanced_chunking(
                    documents=documents,
                    content_type=content_type,
                    use_hierarchical=(self.config.chunking_strategy == "hierarchical")
                )
            else:
                # Use simple indexing
                self.indexer.index_documents(documents, content_type)
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    async def index_documents_async(self, 
                                  documents: List[Document],
                                  content_type: ContentType = ContentType.PLAIN_TEXT) -> None:
        """Asynchronously index documents."""
        await self.indexing_pipeline.index_documents_async(documents, content_type)
    
    def get_query_engine(self, **kwargs) -> RetrieverQueryEngine:
        """
        Get a query engine using the modern retrieval pipeline.
        
        Returns:
            RetrieverQueryEngine configured with modern components
        """
        if self.config.enable_enhanced_retrieval:
            retriever = self.modern_retriever
        else:
            # Fallback to basic retriever
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core import GPTVectorStoreIndex, StorageContext
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            
            vector_store = WeaviateVectorStore(
                weaviate_client=self.indexer.client,
                class_prefix=self.config.class_name.replace("_Node", "")
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = GPTVectorStoreIndex([], storage_context=storage_context)
            retriever = VectorIndexRetriever(index, similarity_top_k=self.config.final_top_k)
        
        return RetrieverQueryEngine.from_args(retriever, **kwargs)
    
    def get_streaming_query_engine(self, **kwargs):
        """Get streaming query engine with real-time step display."""
        base_query_engine = self.get_query_engine(**kwargs)
        
        if hasattr(self, 'fast_query_processor') and self.config.enable_streaming:
            return StreamingQueryEngine(
                base_query_engine=base_query_engine,
                fast_processor=self.fast_query_processor,
                enable_streaming=True
            )
        else:
            return base_query_engine
    
    async def enhanced_query(self, 
                           query: str,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using all modern RAG enhancements.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Enhanced query result with routing and processing information
        """
        result = {
            'original_query': query,
            'enhanced_query': query,
            'routing_decision': None,
            'hyde_components': None
        }
        
        try:
            # Step 1: Query routing and classification
            if self.config.enable_query_routing:
                processing_plan = await self.query_processor.process_query(query, context)
                result['routing_decision'] = processing_plan['routing_decision']
                
                # Extract domain context
                if not context:
                    context = {}
                context.update({
                    'intent': processing_plan['routing_decision'].search_params.get('intent'),
                    'complexity': processing_plan['routing_decision'].search_params.get('complexity')
                })
            
            # Step 2: HyDE query enhancement
            if self.config.enable_hyde:
                # Determine source types from routing
                source_types = ['technical']
                if result['routing_decision']:
                    source_mapping = {
                        'slack': 'conversational',
                        'wiki': 'wiki',
                        'web': 'technical'
                    }
                    source_types = [
                        source_mapping.get(source.value, 'technical')
                        for source in result['routing_decision'].data_sources
                    ]
                
                hyde_components = await self.hyde_integration.enhance_search_query(
                    query, source_types, context.get('domain', 'janelia')
                )
                result['hyde_components'] = hyde_components
                result['enhanced_query'] = hyde_components.get('combined_query', query)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            return result
    
    def get_latest_timestamp(self, source: Optional[str] = None) -> float:
        """Get the latest timestamp from indexed documents."""
        return self.indexer.get_latest_timestamp(source)
    
    def close(self):
        """Close connections and cleanup resources."""
        self.indexer.close()

class JaneliaGPTModernAdapter:
    """
    Adapter class to integrate modern RAG components with existing JaneliaGPT code.
    
    This provides backward compatibility while enabling modern features.
    """
    
    def __init__(self, 
                 weaviate_url: str = "http://localhost:8777",
                 class_prefix: str = "Janelia",
                 enable_modern_features: bool = True):
        """Initialize the adapter."""
        self.weaviate_url = weaviate_url
        self.class_prefix = class_prefix
        self.enable_modern_features = enable_modern_features
        
        if enable_modern_features:
            # Initialize modern system
            config = ModernRAGConfig(
                weaviate_url=weaviate_url,
                class_name=f"{class_prefix}_Node"
            )
            self.modern_rag = ModernRAGSystem(config)
        else:
            # Fallback to legacy system
            from weaviate_indexer import Indexer
            self.legacy_indexer = Indexer(
                weaviate_url=weaviate_url,
                class_prefix=class_prefix,
                delete_database=False
            )
    
    def get_query_engine(self, **kwargs):
        """Get query engine - uses modern or legacy based on configuration."""
        if self.enable_modern_features:
            return self.modern_rag.get_query_engine(**kwargs)
        else:
            # Legacy query engine creation
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core import GPTVectorStoreIndex, StorageContext
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            import weaviate
            
            client = weaviate.Client(self.weaviate_url)
            vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=self.class_prefix)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = GPTVectorStoreIndex([], storage_context=storage_context)
            retriever = VectorIndexRetriever(index, similarity_top_k=10)
            return RetrieverQueryEngine.from_args(retriever, **kwargs)
    
    def index_documents(self, documents: List[Document], **kwargs):
        """Index documents - uses modern or legacy based on configuration."""
        if self.enable_modern_features:
            # Determine content type from kwargs or document metadata
            content_type = kwargs.get('content_type', ContentType.PLAIN_TEXT)
            self.modern_rag.index_documents(documents, content_type)
        else:
            self.legacy_indexer.index(documents)
    
    def get_latest_timestamp(self, source: Optional[str] = None) -> float:
        """Get latest timestamp - uses modern or legacy based on configuration."""
        if self.enable_modern_features:
            return self.modern_rag.get_latest_timestamp(source)
        else:
            if source:
                return self.legacy_indexer.get_latest_slack_timestamp()
            else:
                return self.legacy_indexer.get_latest_timestamp()

# Convenience functions for easy integration
def create_modern_rag_system(weaviate_url: str = "http://localhost:8777",
                            class_name: str = "Janelia_Node",
                            **config_kwargs) -> ModernRAGSystem:
    """Create a modern RAG system with sensible defaults."""
    config = ModernRAGConfig(
        weaviate_url=weaviate_url,
        class_name=class_name,
        **config_kwargs
    )
    return ModernRAGSystem(config)

def create_janelia_adapter(weaviate_url: str = "http://localhost:8777",
                          class_prefix: str = "Janelia",
                          enable_modern: bool = True) -> JaneliaGPTModernAdapter:
    """Create JaneliaGPT adapter with modern features."""
    return JaneliaGPTModernAdapter(
        weaviate_url=weaviate_url,
        class_prefix=class_prefix,
        enable_modern_features=enable_modern
    )

def migrate_to_modern_rag(legacy_weaviate_url: str,
                         legacy_class_prefix: str,
                         modern_config: Optional[ModernRAGConfig] = None) -> ModernRAGSystem:
    """
    Helper function to migrate from legacy to modern RAG system.
    
    This function can be used to gradually migrate existing data and queries
    to the modern system while maintaining compatibility.
    """
    if not modern_config:
        modern_config = ModernRAGConfig(
            weaviate_url=legacy_weaviate_url,
            class_name=f"{legacy_class_prefix}_Node"
        )
    
    modern_system = ModernRAGSystem(modern_config)
    
    # Create schema (this will coexist with legacy schema)
    modern_system.create_schema(delete_existing=False)
    
    logger.info("Modern RAG system created. Legacy system can continue to operate in parallel.")
    return modern_system

# Example usage and testing
if __name__ == "__main__":
    # Example: Create and test modern RAG system
    async def test_modern_rag():
        """Test the modern RAG system."""
        # Create system
        modern_rag = create_modern_rag_system()
        
        # Create schema
        modern_rag.create_schema()
        
        # Test document indexing
        test_docs = [
            Document(
                text="This is a test document about machine learning and neural networks.",
                metadata={"title": "ML Test", "source": "test"}
            )
        ]
        
        modern_rag.index_documents(test_docs, ContentType.PLAIN_TEXT)
        
        # Test enhanced query
        result = await modern_rag.enhanced_query(
            "How do neural networks work?",
            context={"domain": "machine_learning"}
        )
        
        print("Enhanced query result:", result)
        
        # Test query engine
        query_engine = modern_rag.get_query_engine()
        response = query_engine.query("What is machine learning?")
        print("Query response:", response)
        
        modern_rag.close()
    
    # Run test
    asyncio.run(test_modern_rag())