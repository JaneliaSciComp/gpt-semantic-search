# Modern RAG System for JaneliaGPT

This document describes the modern RAG (Retrieval-Augmented Generation) enhancements for JaneliaGPT, implementing state-of-the-art techniques for improved search relevance and response quality.

## Overview

The modern RAG system includes:

- **Advanced Chunking Strategies**: Hierarchical, syntax-aware, and adaptive chunking
- **Multi-Vector Embeddings**: Specialized embeddings for different content types  
- **HyDE Query Transformation**: Hypothetical document embeddings for better retrieval
- **Agentic Query Routing**: Intelligent query classification and routing
- **Enhanced Retrieval Pipeline**: Multi-stage retrieval with reranking

## Architecture

```
User Query
    ↓
Query Classification & Routing (Agentic)
    ↓
HyDE Query Enhancement (Optional)
    ↓
Multi-Stage Retrieval Pipeline:
  - Dense Vector Search
  - Sparse/BM25 Search  
  - Hybrid Fusion (RRF)
  - Cross-Encoder Reranking
  - Contextual Expansion
    ↓
Response Generation
```

## Key Features

### 1. Advanced Chunking Strategies

#### Hierarchical Chunking
Creates parent-child relationships between document sections:

```python
from chunking.advanced_chunkers import HierarchicalChunker, ContentType

chunker = HierarchicalChunker(max_chunk_size=512, overlap_size=50)
chunks = chunker.chunk(text, doc_id="doc1", content_type=ContentType.MARKDOWN)
```

**Benefits:**
- Preserves document structure
- Enables contextual retrieval
- Better handling of long documents

#### Syntax-Aware Chunking
Respects content structure (code functions, markdown headers, etc.):

```python
from chunking.advanced_chunkers import SyntaxAwareChunker

chunker = SyntaxAwareChunker(target_chunk_size=512)
chunks = chunker.chunk(code_text, doc_id="code1", content_type=ContentType.CODE)
```

**Benefits:**
- Code-aware splitting at function boundaries
- Markdown header structure preservation
- Message-based chunking for Slack conversations

#### Adaptive Chunking
Dynamically adjusts chunk size based on content characteristics:

```python
from chunking.advanced_chunkers import AdaptiveChunker

chunker = AdaptiveChunker(base_chunk_size=512)
chunks = chunker.chunk(text, doc_id="doc1", content_type=ContentType.TECHNICAL)
```

**Benefits:**
- Smaller chunks for dense technical content
- Larger chunks for narrative content
- Content density analysis

### 2. Multi-Vector Embeddings

Uses multiple embedding models for different aspects of content:

- **Dense**: Main semantic embeddings (text-embedding-3-large)
- **Sparse**: Compact embeddings for efficiency (text-embedding-3-small)
- **Code**: Specialized for technical content
- **Conversational**: Optimized for Slack messages
- **Summary**: Document overview embeddings

```python
from indexing.modern_weaviate_indexer import ModernWeaviateIndexer, EmbeddingType

indexer = ModernWeaviateIndexer(
    weaviate_url="http://localhost:8777",
    class_name="Janelia_Node",
    embedding_configs={
        EmbeddingType.DENSE: dense_config,
        EmbeddingType.CODE: code_config
    }
)
```

### 3. HyDE Query Transformation

Generates hypothetical documents that would answer the query, then uses those for retrieval:

```python
from chunking.hyde_query_transformer import HyDEIntegration

hyde = HyDEIntegration()
enhanced_query = await hyde.enhance_search_query(
    "How to configure neural networks?",
    source_types=['technical', 'wiki'],
    domain_context='machine_learning'
)
```

**Benefits:**
- Better retrieval for conceptual queries
- Handles vocabulary mismatch between query and documents
- Multiple strategies for different content types

### 4. Agentic Query Routing

Intelligent query analysis and routing:

```python
from agents.query_router import AgenticQueryProcessor

processor = AgenticQueryProcessor()
plan = await processor.process_query(
    "Why is my training failing with CUDA errors?",
    context={'domain': 'machine_learning'}
)
```

**Query Classification:**
- **Intent**: technical_question, troubleshooting, comparison, etc.
- **Complexity**: simple, moderate, complex
- **Data Sources**: slack, wiki, web, all
- **Processing Strategy**: direct_search, technical_deep_dive, etc.

### 5. Enhanced Retrieval Pipeline

Multi-stage retrieval with sophisticated ranking:

```python
from retrieval.enhanced_retrieval import EnhancedRetrievalPipeline

pipeline = EnhancedRetrievalPipeline(indexer, config)
results = await pipeline.retrieve("How to debug neural network training?")
```

**Pipeline Stages:**
1. **Dense Retrieval**: Semantic vector search
2. **Sparse Retrieval**: BM25/keyword search  
3. **Hybrid Fusion**: Reciprocal Rank Fusion
4. **Cross-Encoder Reranking**: LLM-based relevance scoring
5. **Contextual Expansion**: Add hierarchical context
6. **Temporal Boosting**: Favor recent content

## Installation and Setup

### 1. Update Dependencies

The modern system requires updated versions:

```bash
# Update pixi.toml
weaviate-client = "==4.9.3"  # (updated from 3.26.2)

# Update docker-compose.yaml  
image: semitechnologies/weaviate:1.26.4  # (updated from 1.24.4)
```

### 2. Environment Setup

```bash
# Install dependencies
pixi install

# Start updated Weaviate
docker compose up weaviate -d

# Verify Weaviate is running
curl http://localhost:8777/v1/.well-known/ready
```

### 3. Initialize Modern RAG System

```python
from modern_rag_integration import create_modern_rag_system

# Create system with defaults
modern_rag = create_modern_rag_system(
    weaviate_url="http://localhost:8777",
    class_name="Janelia_Node"
)

# Create schema
modern_rag.create_schema()
```

## Usage Examples

### Basic Integration

Replace existing query engine creation:

```python
# OLD: Legacy approach
from weaviate_indexer import Indexer
indexer = Indexer(weaviate_url, class_prefix, False)

# NEW: Modern approach
from modern_rag_integration import create_janelia_adapter
adapter = create_janelia_adapter(
    weaviate_url="http://localhost:8777",
    enable_modern=True
)
query_engine = adapter.get_query_engine()
```

### Advanced Document Indexing

```python
from llama_index.core import Document
from chunking.advanced_chunkers import ContentType

# Index with advanced chunking
documents = [
    Document(text=wiki_content, metadata={"source": "wiki", "title": "Setup Guide"}),
    Document(text=code_content, metadata={"source": "code", "title": "API Reference"})
]

# Index wiki content with hierarchical chunking
modern_rag.index_documents(
    documents[:1], 
    content_type=ContentType.MARKDOWN,
    use_advanced_chunking=True
)

# Index code with syntax-aware chunking  
modern_rag.index_documents(
    documents[1:],
    content_type=ContentType.CODE,
    use_advanced_chunking=True
)
```

### Enhanced Query Processing

```python
# Process query with all enhancements
query = "How do I fix CUDA out of memory errors during training?"

# Get enhanced query components
enhanced_result = await modern_rag.enhanced_query(
    query,
    context={'domain': 'machine_learning', 'urgency': 'high'}
)

print(f"Original: {enhanced_result['original_query']}")
print(f"Enhanced: {enhanced_result['enhanced_query']}")
print(f"Routing: {enhanced_result['routing_decision'].strategy}")

# Use enhanced query engine
query_engine = modern_rag.get_query_engine()
response = query_engine.query(enhanced_result['enhanced_query'])
```

### Migration from Legacy System

```python
from modern_rag_integration import migrate_to_modern_rag

# Migrate gradually - both systems can coexist
modern_system = migrate_to_modern_rag(
    legacy_weaviate_url="http://localhost:8777",
    legacy_class_prefix="Janelia"
)

# Modern system uses "Janelia_Node_Modern" class
# Legacy system continues using "Janelia_Node" class
```

## Configuration Options

### ModernRAGConfig

```python
from modern_rag_integration import ModernRAGConfig

config = ModernRAGConfig(
    # Weaviate settings
    weaviate_url="http://localhost:8777",
    class_name="Janelia_Node",
    
    # Model settings
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-large",
    temperature=0.3,
    
    # Chunking settings
    chunking_strategy="hierarchical",  # or "syntax_aware", "adaptive"
    max_chunk_size=512,
    min_chunk_size=100,
    overlap_size=50,
    
    # Features toggles
    enable_multi_vector=True,
    enable_hyde=True,
    enable_query_routing=True,
    enable_enhanced_retrieval=True,
    enable_reranking=True,
    
    # Retrieval settings
    dense_top_k=50,
    sparse_top_k=30,
    final_top_k=10,
    rerank_threshold=0.5
)
```

### Content Type Mapping

Map your data sources to content types for optimal processing:

```python
content_type_mapping = {
    'slack': ContentType.SLACK,
    'wiki': ContentType.MARKDOWN,  
    'web': ContentType.WEB,
    'code': ContentType.CODE,
    'docs': ContentType.PLAIN_TEXT
}
```

## Performance Considerations

### Memory Usage
- Multi-vector embeddings increase storage by ~3x
- Use sparse embeddings for memory-constrained environments
- Configure `enable_multi_vector=False` to use single embeddings

### Latency
- HyDE adds ~2-3 seconds for complex queries
- Cross-encoder reranking adds ~1-2 seconds  
- Disable features for faster responses:
  ```python
  config = ModernRAGConfig(
      enable_hyde=False,           # Faster query processing
      enable_reranking=False,      # Faster retrieval
      enable_query_routing=False   # Simpler routing
  )
  ```

### Cost Optimization
- Use smaller models for cost reduction:
  ```python
  config = ModernRAGConfig(
      llm_model="gpt-4o-mini",           # vs gpt-4o
      embedding_model="text-embedding-3-small"  # vs text-embedding-3-large
  )
  ```

## Monitoring and Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Component-specific loggers
logging.getLogger('chunking.advanced_chunkers').setLevel(logging.INFO)
logging.getLogger('agents.query_router').setLevel(logging.DEBUG)
logging.getLogger('retrieval.enhanced_retrieval').setLevel(logging.INFO)
```

### Query Analysis

```python
# Analyze query processing
enhanced_result = await modern_rag.enhanced_query(query)
routing = enhanced_result['routing_decision']

print(f"Intent: {routing.search_params['intent']}")
print(f"Complexity: {routing.search_params['complexity']}")
print(f"Strategy: {routing.strategy}")
print(f"Data Sources: {[s.value for s in routing.data_sources]}")
print(f"Confidence: {routing.confidence}")
```

### Performance Metrics

```python
# Measure retrieval performance
import time

start = time.time()
results = await pipeline.retrieve(query)
retrieval_time = time.time() - start

print(f"Retrieved {len(results)} results in {retrieval_time:.2f}s")

# Check result quality
for i, result in enumerate(results[:3]):
    print(f"Result {i+1}: Score={result.score:.3f}")
    print(f"Content: {result.node.text[:100]}...")
```

## Best Practices

### 1. Content Type Classification
- Always specify appropriate content types during indexing
- Use `ContentType.CODE` for code repositories
- Use `ContentType.SLACK` for conversation data
- Use `ContentType.MARKDOWN` for wiki/documentation

### 2. Query Optimization
- Enable HyDE for conceptual/research queries
- Disable HyDE for simple lookup queries
- Use query routing for complex multi-step queries

### 3. Index Management
- Rebuild indexes when upgrading chunking strategies
- Monitor index size growth with multi-vector embeddings
- Use incremental indexing for large datasets

### 4. Error Handling
```python
try:
    results = await modern_rag.enhanced_query(query)
except Exception as e:
    logger.error(f"Modern RAG failed: {e}")
    # Fallback to legacy system
    results = legacy_query_engine.query(query)
```

## Troubleshooting

### Common Issues

1. **Weaviate Connection Errors**
   ```bash
   # Check Weaviate status
   docker compose ps weaviate
   curl http://localhost:8777/v1/.well-known/live
   ```

2. **Memory Issues with Multi-Vector**
   ```python
   # Reduce embedding dimensions or disable multi-vector
   config.enable_multi_vector = False
   ```

3. **Slow Query Performance**
   ```python
   # Reduce retrieval stages
   config.enable_hyde = False
   config.enable_reranking = False
   ```

4. **Schema Migration Issues**
   ```python
   # Create new schema instead of updating
   modern_rag.create_schema(delete_existing=True)
   ```

### Getting Help

- Check logs for detailed error messages
- Use debug mode for step-by-step processing info
- Monitor Weaviate logs: `docker compose logs weaviate`
- Verify OpenAI API key and rate limits

## Migration Guide

### Phase 1: Setup (No Breaking Changes)
1. Update dependencies in `pixi.toml`
2. Update Weaviate in `docker-compose.yaml`
3. Install modern RAG components
4. Test with small dataset

### Phase 2: Parallel Operation
1. Create modern schema alongside legacy
2. Index new documents using modern system
3. A/B test query performance
4. Monitor resource usage

### Phase 3: Full Migration
1. Migrate all existing data to modern schema
2. Update query engine creation
3. Remove legacy components
4. Optimize configuration for production

## Conclusion

The modern RAG system significantly improves search relevance and response quality through:

- **Better Chunking**: Preserves document structure and context
- **Smarter Retrieval**: Multi-stage pipeline with advanced ranking
- **Query Intelligence**: Understands intent and routes appropriately  
- **Multi-Modal Embeddings**: Specialized for different content types

The system is designed for backward compatibility and gradual migration, allowing you to adopt modern features incrementally while maintaining existing functionality.