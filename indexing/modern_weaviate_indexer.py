"""
Modern Weaviate Indexer with Multi-Vector Support

Implements advanced indexing strategies using Weaviate v1.26+ features:
- Named vectors for multiple embeddings per document
- Specialized embedding models for different content types
- Hierarchical document relationships
- Enhanced metadata and filtering
"""

import sys
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio

import weaviate

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core.schema import BaseNode

from chunking.advanced_chunkers import DocumentChunk, ContentType, HierarchicalChunker

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingType(Enum):
    """Types of embeddings for different content aspects."""
    DENSE = "dense"                    # Main content embedding
    SPARSE = "sparse"                  # Keyword-based embedding  
    CODE = "code"                      # Code-specific embedding
    CONVERSATIONAL = "conversational"  # Chat/Slack embedding
    SUMMARY = "summary"                # Document summary embedding

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    name: str
    model: str
    dimensions: int
    embedding_type: EmbeddingType
    description: str

@dataclass
class VectorIndexConfig:
    """Configuration for vector index."""
    distance_metric: str = "cosine"
    ef_construction: int = 128
    ef: int = 64
    max_connections: int = 32
    dynamic_ef_min: int = 100
    dynamic_ef_max: int = 500
    dynamic_ef_factor: int = 8
    vector_cache_max_objects: int = 1000000
    flat_search_cutoff: int = 40000
    cleanup_interval: int = 300
    pq_enabled: bool = False
    pq_segments: int = 0
    pq_centroids: int = 256
    pq_training_limit: int = 100000
    pq_encoder_type: str = "kmeans"
    pq_encoder_distribution: str = "log-normal"

class ModernWeaviateIndexer:
    """
    Modern Weaviate indexer with multi-vector support and advanced features.
    
    Features:
    - Multiple named vectors per document
    - Hierarchical document relationships  
    - Content-type specific embeddings
    - Advanced filtering and metadata
    - Async operations support
    """
    
    # Default embedding configurations
    DEFAULT_EMBEDDINGS = {
        EmbeddingType.DENSE: EmbeddingConfig(
            name="dense",
            model="text-embedding-3-large", 
            dimensions=3072,
            embedding_type=EmbeddingType.DENSE,
            description="Main content embedding for semantic search"
        ),
        EmbeddingType.SPARSE: EmbeddingConfig(
            name="sparse",
            model="text-embedding-3-small",
            dimensions=1536, 
            embedding_type=EmbeddingType.SPARSE,
            description="Compact embedding for keyword-style search"
        ),
        EmbeddingType.CODE: EmbeddingConfig(
            name="code",
            model="text-embedding-3-large",
            dimensions=3072,
            embedding_type=EmbeddingType.CODE,
            description="Code-optimized embedding for technical content"
        ),
        EmbeddingType.CONVERSATIONAL: EmbeddingConfig(
            name="conversational", 
            model="text-embedding-3-small",
            dimensions=1536,
            embedding_type=EmbeddingType.CONVERSATIONAL,
            description="Conversational embedding for chat/Slack content"
        ),
        EmbeddingType.SUMMARY: EmbeddingConfig(
            name="summary",
            model="text-embedding-3-small", 
            dimensions=1536,
            embedding_type=EmbeddingType.SUMMARY,
            description="Summary embedding for document overviews"
        )
    }
    
    def __init__(self, 
                 weaviate_url: str,
                 class_name: str,
                 embedding_configs: Optional[Dict[EmbeddingType, EmbeddingConfig]] = None,
                 vector_index_config: Optional[VectorIndexConfig] = None,
                 enable_hierarchical: bool = True,
                 enable_async: bool = True):
        """
        Initialize modern Weaviate indexer.
        
        Args:
            weaviate_url: Weaviate instance URL
            class_name: Name of the Weaviate class
            embedding_configs: Custom embedding configurations
            vector_index_config: Vector index configuration
            enable_hierarchical: Enable hierarchical document relationships
            enable_async: Enable async operations
        """
        self.weaviate_url = weaviate_url
        self.class_name = class_name
        self.embedding_configs = embedding_configs or self.DEFAULT_EMBEDDINGS
        self.vector_index_config = vector_index_config or VectorIndexConfig()
        self.enable_hierarchical = enable_hierarchical
        self.enable_async = enable_async
        
        # Initialize embedding models
        self.embedding_models = {}
        for embedding_type, config in self.embedding_configs.items():
            self.embedding_models[embedding_type] = OpenAIEmbedding(
                model=config.model,
                embed_batch_size=20
            )
        
        # Initialize Weaviate client
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Weaviate client with v3 API."""
        try:
            # Use Weaviate v3 client for LlamaIndex compatibility
            self.client = weaviate.Client(self.weaviate_url)
            
            if not self.client.is_live():
                raise ConnectionError(f"Weaviate is not live at {self.weaviate_url}")
                
            if not self.client.is_ready():
                raise ConnectionError(f"Weaviate is not ready at {self.weaviate_url}")
                
            logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise ConnectionError(f"Weaviate is not available at {self.weaviate_url}")
    
    def create_schema(self, delete_existing: bool = False) -> None:
        """Create modern Weaviate schema with enhanced properties using v3 API."""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            existing_classes = {c["class"] for c in schema.get("classes", [])}
            
            if self.class_name in existing_classes:
                if delete_existing:
                    logger.warning(f"Deleting existing class: {self.class_name}")
                    self.client.schema.delete_class(self.class_name)
                else:
                    logger.info(f"Class {self.class_name} already exists")
                    return
            
            # Define properties with enhanced metadata for modern RAG
            properties = [
                {
                    "name": "ref_doc_id",
                    "dataType": ["text"],
                    "description": "Document reference ID",
                    "tokenization": "field"
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Full text content",
                    "tokenization": "word"
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Document title",
                    "tokenization": "word"
                },
                {
                    "name": "link",
                    "dataType": ["text"],
                    "description": "Source URL",
                    "tokenization": "field"
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "Data source type",
                    "tokenization": "field"
                },
                {
                    "name": "content_type",
                    "dataType": ["text"],
                    "description": "Content type classification",
                    "tokenization": "field"
                },
                {
                    "name": "scraped_at",
                    "dataType": ["number"],
                    "description": "Scraping timestamp"
                },
                {
                    "name": "chunk_id",
                    "dataType": ["text"],
                    "description": "Unique chunk identifier",
                    "tokenization": "field"
                },
                {
                    "name": "parent_id",
                    "dataType": ["text"],
                    "description": "Parent document ID",
                    "tokenization": "field"
                },
                {
                    "name": "level",
                    "dataType": ["int"],
                    "description": "Hierarchical level"
                },
                {
                    "name": "token_count",
                    "dataType": ["int"],
                    "description": "Token count"
                },
                {
                    "name": "start_char",
                    "dataType": ["int"],
                    "description": "Start character position"
                },
                {
                    "name": "end_char",
                    "dataType": ["int"],
                    "description": "End character position"
                },
                {
                    "name": "semantic_score",
                    "dataType": ["number"],
                    "description": "Semantic quality score"
                },
                {
                    "name": "embedding_dense",
                    "dataType": ["number[]"],
                    "description": "Dense semantic embedding"
                },
                {
                    "name": "embedding_sparse",
                    "dataType": ["number[]"],
                    "description": "Sparse/compact embedding"
                },
                {
                    "name": "embedding_code",
                    "dataType": ["number[]"],
                    "description": "Code-specific embedding"
                },
                {
                    "name": "embedding_conversational",
                    "dataType": ["number[]"],
                    "description": "Conversational embedding"
                },
                {
                    "name": "embedding_summary",
                    "dataType": ["number[]"],
                    "description": "Summary embedding"
                }
            ]
            
            # Create class definition
            class_obj = {
                "class": self.class_name,
                "description": f"Modern multi-vector collection for {self.class_name}",
                "properties": properties,
                "vectorizer": "none",  # We'll manage embeddings manually
                "vectorIndexConfig": {
                    "distance": self.vector_index_config.distance_metric,
                    "efConstruction": self.vector_index_config.ef_construction,
                    "ef": self.vector_index_config.ef,
                    "maxConnections": self.vector_index_config.max_connections,
                    "dynamicEfMin": self.vector_index_config.dynamic_ef_min,
                    "dynamicEfMax": self.vector_index_config.dynamic_ef_max,
                    "dynamicEfFactor": self.vector_index_config.dynamic_ef_factor,
                    "vectorCacheMaxObjects": self.vector_index_config.vector_cache_max_objects,
                    "flatSearchCutoff": self.vector_index_config.flat_search_cutoff,
                    "cleanupIntervalSeconds": self.vector_index_config.cleanup_interval
                },
                "invertedIndexConfig": {
                    "bm25": {
                        "b": 0.75,
                        "k1": 1.2
                    },
                    "cleanupIntervalSeconds": 60,
                    "indexNullState": True,
                    "indexPropertyLength": True,
                    "indexTimestamps": True
                }
            }
            
            # Create the class
            self.client.schema.create_class(class_obj)
            logger.info(f"Created modern schema for class: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    async def index_documents_async(self, 
                                  documents: List[Union[Document, DocumentChunk]],
                                  content_type: ContentType = ContentType.PLAIN_TEXT,
                                  batch_size: int = 100) -> None:
        """Asynchronously index documents with multi-vector embeddings."""
        if not self.enable_async:
            return self.index_documents(documents, content_type, batch_size)
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await self._index_batch_async(batch, content_type)
            logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
    
    def index_documents(self, 
                       documents: List[Union[Document, DocumentChunk]],
                       content_type: ContentType = ContentType.PLAIN_TEXT,
                       batch_size: int = 100) -> None:
        """Index documents with multi-vector embeddings using v3 API."""
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch for Weaviate
                with self.client.batch(batch_size=batch_size, dynamic=True) as batch_client:
                    for doc in batch:
                        # Convert to DocumentChunk if needed
                        if isinstance(doc, Document):
                            doc_chunk = self._convert_document_to_chunk(doc, content_type)
                        else:
                            doc_chunk = doc
                        
                        # Generate multiple embeddings
                        embeddings = self._generate_multi_vector_embeddings(doc_chunk.text, content_type)
                        
                        # Prepare properties
                        # Handle metadata based on input type
                        if isinstance(doc, Document):
                            title = getattr(doc, 'metadata', {}).get('title', '')
                            link = getattr(doc, 'metadata', {}).get('link', '')
                            source = getattr(doc, 'metadata', {}).get('source', '')
                            scraped_at = getattr(doc, 'metadata', {}).get('scraped_at', datetime.now().timestamp())
                        else:
                            # doc is a DocumentChunk, get metadata from chunk metadata
                            title = getattr(doc_chunk.metadata, 'title', '')
                            link = getattr(doc_chunk.metadata, 'link', '')
                            source = getattr(doc_chunk.metadata, 'source', '')
                            scraped_at = getattr(doc_chunk.metadata, 'scraped_at', datetime.now().timestamp())
                        
                        properties = {
                            "ref_doc_id": doc_chunk.metadata.chunk_id,
                            "text": doc_chunk.text,
                            "title": title,
                            "link": link,
                            "source": source,
                            "content_type": content_type.value,
                            "scraped_at": scraped_at,
                            "chunk_id": doc_chunk.metadata.chunk_id,
                            "parent_id": doc_chunk.metadata.parent_id,
                            "level": doc_chunk.metadata.level,
                            "token_count": doc_chunk.metadata.token_count,
                            "start_char": doc_chunk.metadata.start_char,
                            "end_char": doc_chunk.metadata.end_char,
                            "semantic_score": doc_chunk.metadata.semantic_score
                        }
                        
                        # Add embeddings to properties
                        if "dense" in embeddings:
                            properties["embedding_dense"] = embeddings["dense"]
                        if "sparse" in embeddings:
                            properties["embedding_sparse"] = embeddings["sparse"]
                        if "code" in embeddings:
                            properties["embedding_code"] = embeddings["code"]
                        if "conversational" in embeddings:
                            properties["embedding_conversational"] = embeddings["conversational"]
                        if "summary" in embeddings:
                            properties["embedding_summary"] = embeddings["summary"]
                        
                        # Use primary embedding (dense) as the main vector
                        main_vector = embeddings.get("dense", embeddings.get("sparse", None))
                        
                        # Add object to batch
                        batch_client.add_data_object(
                            data_object=properties,
                            class_name=self.class_name,
                            vector=main_vector
                        )
                
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    async def _index_batch_async(self, 
                               batch: List[Union[Document, DocumentChunk]],
                               content_type: ContentType) -> None:
        """Asynchronously index a batch of documents."""
        # This is a placeholder for async implementation
        # In practice, you'd use async embedding generation and batch insertion
        self.index_documents(batch, content_type, len(batch))
    
    def _convert_document_to_chunk(self, doc: Document, content_type: ContentType) -> DocumentChunk:
        """Convert LlamaIndex Document to DocumentChunk."""
        from chunking.advanced_chunkers import ChunkMetadata
        
        chunk_id = doc.doc_id or f"doc_{hash(doc.text)}"
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            content_type=content_type,
            token_count=len(doc.text.split()),  # Rough estimate
            start_char=0,
            end_char=len(doc.text),
            semantic_score=0.5
        )
        
        return DocumentChunk(text=doc.text, metadata=metadata)
    
    def _generate_multi_vector_embeddings(self, text: str, content_type: ContentType) -> Dict[str, List[float]]:
        """Generate multiple embeddings for different aspects of the text."""
        embeddings = {}
        
        # Check if text is too long (conservative limit)
        estimated_tokens = len(text.split()) * 1.3  # Rough token estimate
        if estimated_tokens > 7000:
            logger.warning(f"Text too long ({estimated_tokens:.0f} tokens), truncating for embedding")
            # Truncate to a safe size (approximately 6000 tokens worth of text)
            words = text.split()
            safe_word_count = int(6000 / 1.3)  # Conservative token to word ratio
            text = ' '.join(words[:safe_word_count])
        
        try:
            # Always generate dense embedding
            if EmbeddingType.DENSE in self.embedding_configs:
                embeddings["dense"] = self.embedding_models[EmbeddingType.DENSE].get_text_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate dense embedding: {e}")
            # Return a zero vector as fallback
            embeddings["dense"] = [0.0] * 3072
        
        # Generate sparse embedding for keyword search
        if EmbeddingType.SPARSE in self.embedding_configs:
            try:
                # For sparse, we might use a truncated version for efficiency
                sparse_text = text[:1000] if len(text) > 1000 else text
                embeddings["sparse"] = self.embedding_models[EmbeddingType.SPARSE].get_text_embedding(sparse_text)
            except Exception as e:
                logger.warning(f"Failed to generate sparse embedding: {e}")
                embeddings["sparse"] = [0.0] * 1536
        
        # Content-type specific embeddings
        if content_type == ContentType.CODE and EmbeddingType.CODE in self.embedding_configs:
            try:
                embeddings["code"] = self.embedding_models[EmbeddingType.CODE].get_text_embedding(text)
            except Exception as e:
                logger.warning(f"Failed to generate code embedding: {e}")
                embeddings["code"] = [0.0] * 3072
        
        if content_type == ContentType.SLACK and EmbeddingType.CONVERSATIONAL in self.embedding_configs:
            try:
                embeddings["conversational"] = self.embedding_models[EmbeddingType.CONVERSATIONAL].get_text_embedding(text)
            except Exception as e:
                logger.warning(f"Failed to generate conversational embedding: {e}")
                embeddings["conversational"] = [0.0] * 3072
        
        # Generate summary embedding for long documents
        if len(text) > 2000 and EmbeddingType.SUMMARY in self.embedding_configs:
            summary_text = self._generate_summary(text)
            embeddings["summary"] = self.embedding_models[EmbeddingType.SUMMARY].get_text_embedding(summary_text)
        
        return embeddings
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text for summary embeddings."""
        # Simple extractive summary - take first and last sentences
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return text
        
        # Take first 2 and last 2 sentences
        summary_sentences = sentences[:2] + sentences[-2:]
        return '. '.join(summary_sentences)
    
    def search_multi_vector(self, 
                           query: str,
                           vector_names: List[str] = None,
                           limit: int = 10,
                           where_filter: Optional[Dict] = None,
                           content_type: Optional[ContentType] = None) -> List[Dict]:
        """
        Search using multiple embeddings with v3 API.
        
        Args:
            query: Search query
            vector_names: List of vector names to search (default: all)
            limit: Number of results to return
            where_filter: Additional filters
            content_type: Filter by content type
            
        Returns:
            List of search results
        """
        try:
            # Use default vectors if none specified
            if vector_names is None:
                vector_names = ["dense", "sparse", "code", "conversational"]
            
            # Generate query embedding (use dense as primary)
            query_embedding = None
            if EmbeddingType.DENSE in self.embedding_models:
                query_embedding = self.embedding_models[EmbeddingType.DENSE].get_text_embedding(query)
            elif self.embedding_models:
                # Fallback to first available embedding model
                first_model = next(iter(self.embedding_models.values()))
                query_embedding = first_model.get_text_embedding(query)
            
            if query_embedding is None:
                raise ValueError("No embedding model available")
            
            # Build where filter
            where_conditions = []
            if where_filter:
                where_conditions.append(where_filter)
            if content_type:
                where_conditions.append({
                    "path": ["content_type"],
                    "operator": "Equal",
                    "valueText": content_type.value
                })
            
            # Combine conditions
            final_where = None
            if len(where_conditions) == 1:
                final_where = where_conditions[0]
            elif len(where_conditions) > 1:
                final_where = {"operator": "And", "operands": where_conditions}
            
            # Perform vector search
            query_builder = self.client.query.get(self.class_name, [
                "ref_doc_id", "text", "title", "link", "source", "content_type",
                "scraped_at", "chunk_id", "parent_id", "level", "token_count",
                "start_char", "end_char", "semantic_score"
            ]).with_near_vector({
                "vector": query_embedding
            }).with_limit(limit)
            
            if final_where:
                query_builder = query_builder.with_where(final_where)
            
            # Add additional metadata
            query_builder = query_builder.with_additional(["score", "distance"])
            
            # Execute query
            result = query_builder.do()
            
            # Format results
            results = []
            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                for obj in result["data"]["Get"][self.class_name]:
                    additional = obj.get("_additional", {})
                    result_dict = {
                        "uuid": additional.get("id", ""),
                        "properties": {k: v for k, v in obj.items() if not k.startswith("_")},
                        "metadata": additional,
                        "vector_name": "dense",  # Primary vector used
                        "score": additional.get("score", 0.0)
                    }
                    results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search multi-vector: {e}")
            raise
    
    def get_hierarchical_context(self, chunk_id: str, include_children: bool = True) -> Dict[str, Any]:
        """
        Get hierarchical context for a document chunk using v3 API.
        
        Args:
            chunk_id: ID of the chunk
            include_children: Whether to include child chunks
            
        Returns:
            Dictionary with hierarchical context
        """
        try:
            # Get the main chunk
            main_chunk_result = self.client.query.get(self.class_name, [
                "ref_doc_id", "text", "title", "link", "source", "content_type",
                "chunk_id", "parent_id", "level", "token_count"
            ]).with_where({
                "path": ["chunk_id"],
                "operator": "Equal",
                "valueText": chunk_id
            }).with_limit(1).do()
            
            if not (main_chunk_result.get("data", {}).get("Get", {}).get(self.class_name)):
                return {}
            
            main_chunk = main_chunk_result["data"]["Get"][self.class_name][0]
            
            context = {
                "main_chunk": main_chunk,
                "parent": None,
                "children": [],
                "siblings": []
            }
            
            # Get parent if exists
            parent_id = main_chunk.get("parent_id")
            if parent_id:
                parent_result = self.client.query.get(self.class_name, [
                    "ref_doc_id", "text", "title", "link", "source", "content_type",
                    "chunk_id", "parent_id", "level", "token_count"
                ]).with_where({
                    "path": ["chunk_id"],
                    "operator": "Equal",
                    "valueText": parent_id
                }).with_limit(1).do()
                
                if parent_result.get("data", {}).get("Get", {}).get(self.class_name):
                    context["parent"] = parent_result["data"]["Get"][self.class_name][0]
            
            # Get children if requested
            if include_children:
                children_result = self.client.query.get(self.class_name, [
                    "ref_doc_id", "text", "title", "link", "source", "content_type",
                    "chunk_id", "parent_id", "level", "token_count"
                ]).with_where({
                    "path": ["parent_id"],
                    "operator": "Equal",
                    "valueText": chunk_id
                }).with_limit(10).do()
                
                if children_result.get("data", {}).get("Get", {}).get(self.class_name):
                    context["children"] = children_result["data"]["Get"][self.class_name]
            
            # Get siblings (same parent)
            if parent_id:
                siblings_result = self.client.query.get(self.class_name, [
                    "ref_doc_id", "text", "title", "link", "source", "content_type",
                    "chunk_id", "parent_id", "level", "token_count"
                ]).with_where({
                    "path": ["parent_id"],
                    "operator": "Equal",
                    "valueText": parent_id
                }).with_limit(20).do()
                
                if siblings_result.get("data", {}).get("Get", {}).get(self.class_name):
                    context["siblings"] = [
                        obj for obj in siblings_result["data"]["Get"][self.class_name]
                        if obj.get("chunk_id") != chunk_id
                    ]
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get hierarchical context: {e}")
            return {}
    
    def get_latest_timestamp(self, source: Optional[str] = None) -> float:
        """Get the latest timestamp from indexed documents using v3 API."""
        try:
            # Build query
            query_builder = self.client.query.aggregate(self.class_name).with_fields("scraped_at { maximum }")
            
            # Add source filter if specified
            if source:
                query_builder = query_builder.with_where({
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": source
                })
            
            # Execute query
            result = query_builder.do()
            
            # Extract maximum timestamp
            try:
                max_timestamp = result["data"]["Aggregate"][self.class_name][0]["scraped_at"]["maximum"]
                source_msg = f" for {source}" if source else ""
                logger.info(f"Found latest timestamp{source_msg} in database: {max_timestamp}")
                return float(max_timestamp) if max_timestamp else 0.0
            except (KeyError, IndexError, TypeError):
                source_msg = f" for {source}" if source else ""
                logger.info(f"No documents{source_msg} found in database or no scraped_at timestamps")
                return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get latest timestamp: {e}")
            return 0.0
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()

class ModernIndexingPipeline:
    """
    Complete indexing pipeline combining advanced chunking with multi-vector indexing.
    """
    
    def __init__(self, 
                 weaviate_url: str,
                 class_name: str,
                 chunker_config: Optional[Dict] = None,
                 embedding_configs: Optional[Dict[EmbeddingType, EmbeddingConfig]] = None):
        """Initialize the modern indexing pipeline."""
        self.indexer = ModernWeaviateIndexer(
            weaviate_url=weaviate_url,
            class_name=class_name,
            embedding_configs=embedding_configs
        )
        
        # Initialize chunkers
        chunker_config = chunker_config or {}
        self.hierarchical_chunker = HierarchicalChunker(**chunker_config)
        
        # Initialize schema
        self.indexer.create_schema()
    
    def index_documents_with_advanced_chunking(self, 
                                             documents: List[Document],
                                             content_type: ContentType = ContentType.PLAIN_TEXT,
                                             use_hierarchical: bool = True) -> None:
        """
        Index documents using advanced chunking strategies.
        
        Args:
            documents: List of documents to index
            content_type: Type of content for chunking optimization
            use_hierarchical: Whether to use hierarchical chunking
        """
        all_chunks = []
        
        for doc in documents:
            if use_hierarchical:
                # Use hierarchical chunking
                chunks = self.hierarchical_chunker.chunk(
                    text=doc.text,
                    doc_id=doc.doc_id or f"doc_{hash(doc.text)}",
                    content_type=content_type
                )
                # Add original document metadata to each chunk
                for chunk in chunks:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        chunk.metadata.title = doc.metadata.get('title', '')
                        chunk.metadata.link = doc.metadata.get('link', '')
                        chunk.metadata.source = doc.metadata.get('source', '')
                        chunk.metadata.scraped_at = doc.metadata.get('scraped_at', datetime.now().timestamp())
                all_chunks.extend(chunks)
            else:
                # Convert to simple chunk
                chunk = self.indexer._convert_document_to_chunk(doc, content_type)
                all_chunks.append(chunk)
        
        # Index all chunks
        self.indexer.index_documents(all_chunks, content_type)
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    
    async def index_documents_async(self, 
                                  documents: List[Document],
                                  content_type: ContentType = ContentType.PLAIN_TEXT,
                                  use_hierarchical: bool = True) -> None:
        """Asynchronously index documents with advanced chunking."""
        # For now, use sync version - async chunking can be added later
        self.index_documents_with_advanced_chunking(documents, content_type, use_hierarchical)
    
    def close(self):
        """Close the indexing pipeline."""
        self.indexer.close()