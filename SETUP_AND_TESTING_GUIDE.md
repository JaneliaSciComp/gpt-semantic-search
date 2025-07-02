# Modern RAG Setup and Testing Guide

This guide provides step-by-step instructions to set up and test the modern RAG enhancements for JaneliaGPT.

## Quick Start

### 1. Update Dependencies

First, ensure you have the updated dependencies:

```bash
# Update pixi environment
pixi install

# Verify Weaviate version update
grep "weaviate-client" pixi.toml
# Should show: weaviate-client = ">=3.26.2,<4.0.0"
```

### 2. Start Updated Weaviate

```bash
# Stop existing Weaviate
docker compose down weaviate

# Start updated Weaviate (v1.25.5 - compatible with v3 client)
docker compose up weaviate -d

# Verify it's running
curl http://localhost:8777/v1/.well-known/ready
```

### 3. Quick Test

```bash
# Run quick dependency check
pixi run python test_modern_rag.py --quick-test
```

If this passes, you're ready to proceed!

## Full Testing Procedure

### Step 1: Run Comprehensive Tests

```bash
# Run full test suite
pixi run python test_modern_rag.py

# View detailed logs
tail -f modern_rag_test.log
```

This will test:
- ✅ Dependencies and connections
- 🔧 Advanced chunking strategies  
- 🔮 HyDE query transformation
- 🤖 Agentic query routing
- 📚 Multi-vector indexing
- 🔍 Enhanced retrieval pipeline

### Step 2: Test Modern Indexing

After the tests pass, try indexing some data with modern features:

```bash
# Index wiki data with modern RAG
pixi run python index_wiki.py -i ./data/wiki -c Janelia --modern-rag

# Or test with a small sample first
pixi run python index_wiki.py -i ./data/wiki -c JaneliaTest --modern-rag -r
```

### Step 3: Test in Streamlit Interface

1. **Start the application:**
   ```bash
   pixi run streamlit run 1_🔍_Search.py
   ```

2. **Navigate to Settings (⚙️):**
   - Go to http://localhost:8501 and click "⚙️ Settings" in the sidebar
   - You should see a new "Modern RAG Features" section

3. **Enable Modern RAG:**
   - Check "Use Modern RAG" checkbox
   - Configure features you want to test:
     - ✅ Enhanced Retrieval Pipeline
     - ✅ HyDE Query Enhancement  
     - ✅ Agentic Query Routing
     - ✅ Cross-Encoder Reranking
   - Select chunking strategy: "hierarchical"

4. **Test Queries:**
   - Go back to the main search page
   - Try these test queries:
     ```
     How to fix CUDA out of memory errors?
     What's the difference between RNN and LSTM?
     How to implement data augmentation in PyTorch?
     ```

## Expected Results

### With Modern RAG Enabled:

1. **Query Processing Time:** 2-5 seconds (vs 1-2 seconds legacy)
2. **Enhanced Query Analysis:** Look for routing strategy in logs
3. **Better Results:** More relevant and comprehensive responses
4. **Source Diversity:** Results from multiple content types

### Features You Should See:

- **Settings Page:** Modern RAG configuration options
- **Console Logs:** "Using modern RAG query engine" messages  
- **Enhanced Responses:** More structured and comprehensive answers
- **Performance Metrics:** Query routing and enhancement info in logs

## Troubleshooting

### Common Issues

1. **"Modern RAG not available" Error**
   ```bash
   # Check if modern components are installed
   ls chunking/ indexing/ agents/ retrieval/
   
   # If missing, ensure all files were created properly
   python -c "from modern_rag_integration import create_modern_rag_system; print('✅ Modern RAG available')"
   ```

2. **Weaviate Connection Errors**
   ```bash
   # Check Weaviate status
   docker compose ps weaviate
   curl http://localhost:8777/v1/.well-known/live
   
   # Restart if needed
   docker compose restart weaviate
   ```

3. **Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   
   # Or run from project root
   cd /path/to/gpt-semantic-search
   pixi run python test_modern_rag.py
   ```

4. **OpenAI API Issues**
   ```bash
   # Verify API key is set
   echo $OPENAI_API_KEY
   
   # Test API access
   pixi run python -c "from openai import OpenAI; print(OpenAI().models.list().data[0].id)"
   ```

5. **Memory Issues**
   ```bash
   # Disable some features if running low on memory
   # In Settings, uncheck:
   # - HyDE Query Enhancement
   # - Cross-Encoder Reranking
   ```

### Performance Issues

If queries are too slow:

1. **Disable Heavy Features:**
   - Uncheck "HyDE Query Enhancement" (saves ~2-3 seconds)
   - Uncheck "Cross-Encoder Reranking" (saves ~1-2 seconds)

2. **Use Lighter Models:**
   - Change LLM model to "gpt-4o-mini" in Settings
   - Modern system will use more efficient embedding combinations

3. **Reduce Retrieval Scope:**
   - The system automatically optimizes based on enabled features

## Comparison Testing

### A/B Testing Setup

1. **Test Legacy System:**
   ```bash
   # In Settings, uncheck "Use Modern RAG"
   # Run test queries and note performance/quality
   ```

2. **Test Modern System:**
   ```bash
   # In Settings, check "Use Modern RAG"  
   # Run same test queries and compare
   ```

### Metrics to Compare

- **Response Quality:** Relevance and completeness
- **Response Time:** Query processing latency
- **Source Coverage:** Diversity of retrieved sources
- **Context Preservation:** Better understanding of follow-up questions

## Migration Strategy

### Phase 1: Testing (Current)
- ✅ Run comprehensive tests
- ✅ Test with small dataset
- ✅ Compare performance with legacy system

### Phase 2: Parallel Operation
```bash
# Create modern schema alongside legacy
# Legacy uses: "Janelia_Node" 
# Modern uses: "Janelia_Node" (updated schema) or "JaneliaModern_Node"

# Index new data with modern system
pixi run python index_wiki.py -i ./data/wiki -c JaneliaModern --modern-rag

# Test both systems in production
```

### Phase 3: Full Migration
```bash
# Backup existing data
# Re-index all data with modern system  
# Update default settings to use modern RAG
# Monitor performance and user feedback
```

## Advanced Configuration

### Custom Configuration

Create a custom config file `modern_rag_config.py`:

```python
from modern_rag_integration import ModernRAGConfig

# High-performance configuration
HIGH_PERF_CONFIG = ModernRAGConfig(
    enable_hyde=True,
    enable_query_routing=True,
    enable_enhanced_retrieval=True,
    enable_reranking=True,
    chunking_strategy="hierarchical",
    dense_top_k=30,
    sparse_top_k=20,
    final_top_k=8
)

# Fast configuration (lower latency)
FAST_CONFIG = ModernRAGConfig(
    enable_hyde=False,
    enable_query_routing=True,
    enable_enhanced_retrieval=True,
    enable_reranking=False,
    chunking_strategy="adaptive",
    dense_top_k=20,
    sparse_top_k=10,
    final_top_k=5
)

# Memory-efficient configuration
MEMORY_EFFICIENT_CONFIG = ModernRAGConfig(
    enable_multi_vector=False,
    enable_code_embeddings=False,
    enable_conversational_embeddings=False,
    chunking_strategy="syntax_aware"
)
```

### Monitoring

Add monitoring to track performance:

```python
# In your application
import time
import logging

logger = logging.getLogger(__name__)

def log_query_performance(query, response_time, routing_strategy, num_results):
    logger.info(f"PERFORMANCE: query='{query[:50]}...', time={response_time:.2f}s, strategy={routing_strategy}, results={num_results}")
```

## Success Criteria

You'll know the modern RAG system is working correctly when:

1. ✅ **All tests pass** in `test_modern_rag.py`
2. ✅ **Settings page shows** modern RAG options  
3. ✅ **Logs show** "Using modern RAG query engine"
4. ✅ **Queries return** more relevant and comprehensive results
5. ✅ **Enhanced features work:** HyDE, routing, reranking
6. ✅ **No errors** during indexing with `--modern-rag` flag

## Next Steps

After successful testing:

1. **Update CLAUDE.md** with modern RAG commands
2. **Re-index production data** with modern chunking
3. **Monitor user feedback** on response quality  
4. **Optimize configuration** based on usage patterns
5. **Consider expanding** to additional content types

## Getting Help

If you encounter issues:

1. **Check logs:** `modern_rag_test.log` and console output
2. **Run diagnostics:** `python test_modern_rag.py --quick-test`
3. **Verify setup:** Ensure all files exist in `chunking/`, `indexing/`, `agents/`, `retrieval/`
4. **Test incrementally:** Start with basic features, add advanced ones gradually

The modern RAG system is backward compatible, so you can always fall back to the legacy system if needed!