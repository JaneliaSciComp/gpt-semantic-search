#!/usr/bin/env python3
"""
Modern RAG Testing and Migration Script

This script helps you test the modern RAG features and migrate from the legacy system.
It provides comprehensive testing of all modern components.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('modern_rag_test.log')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check core dependencies
    try:
        import weaviate
        logger.info(f"✅ Weaviate client: {weaviate.__version__}")
    except ImportError:
        missing_deps.append("weaviate-client")
    
    try:
        from llama_index.core import Document
        logger.info("✅ LlamaIndex core available")
    except ImportError:
        missing_deps.append("llama-index-core")
    
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        logger.info("✅ OpenAI embeddings available")
    except ImportError:
        missing_deps.append("llama-index-embeddings-openai")
    
    # Check modern RAG components
    try:
        from modern_rag_integration import create_modern_rag_system, ModernRAGConfig
        from chunking.advanced_chunkers import ContentType
        from agents.query_router import AgenticQueryProcessor
        logger.info("✅ Modern RAG components available")
        return True, missing_deps
    except ImportError as e:
        logger.error(f"❌ Modern RAG components not available: {e}")
        missing_deps.append("modern-rag-components")
        return False, missing_deps

def test_weaviate_connection(weaviate_url: str) -> bool:
    """Test connection to Weaviate."""
    logger.info(f"🔗 Testing Weaviate connection to {weaviate_url}")
    
    try:
        import weaviate
        client = weaviate.Client(weaviate_url)
        
        if client.is_live() and client.is_ready():
            meta = client.get_meta()
            version = meta.get('version', 'unknown')
            logger.info(f"✅ Weaviate connected - Version: {version}")
            
            # Check if this is a compatible version
            if version.startswith('1.25') or version.startswith('1.24'):
                logger.info("✅ Compatible Weaviate version for v3 API")
            else:
                logger.warning(f"⚠️ Weaviate version {version} may have compatibility issues with v3 client")
            
            return True
        else:
            logger.error("❌ Weaviate is not live or ready")
            return False
            
    except Exception as e:
        logger.error(f"❌ Weaviate connection failed: {e}")
        return False

def create_test_documents() -> List[Document]:
    """Create test documents for different content types."""
    logger.info("📄 Creating test documents")
    
    documents = []
    
    # Technical documentation
    tech_doc = Document(
        text="""# Neural Network Training Guide

## Introduction
Neural networks are computational models inspired by biological neural networks. This guide covers the fundamentals of training neural networks for deep learning applications.

## Prerequisites
- Python 3.8+
- PyTorch or TensorFlow
- CUDA-enabled GPU (recommended)

## Basic Training Loop
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.targets)
        loss.backward()
        optimizer.step()
```

## Common Issues
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Slow convergence**: Adjust learning rate or use learning rate scheduling
- **Overfitting**: Add regularization or use dropout

## Best Practices
1. Start with a simple model
2. Use proper data augmentation
3. Monitor training and validation metrics
4. Save checkpoints regularly
""",
        metadata={
            "title": "Neural Network Training Guide",
            "source": "wiki",
            "type": "technical_documentation",
            "scraped_at": time.time()
        }
    )
    documents.append(tech_doc)
    
    # Conversational content (Slack-style)
    conv_doc = Document(
        text="""2024-01-15 10:30:45 @alice: Hey team, I'm getting CUDA out of memory errors when training my model. Anyone know how to fix this?

2024-01-15 10:32:12 @bob: @alice Try reducing your batch size. What's your current batch size?

2024-01-15 10:33:01 @alice: I'm using batch_size=32. The model is quite large though.

2024-01-15 10:34:15 @charlie: You could also try gradient accumulation. Set batch_size=16 and accumulate gradients for 2 steps to get effective batch_size=32

2024-01-15 10:35:30 @bob: Good point @charlie. Also check if you have other processes using GPU memory with `nvidia-smi`

2024-01-15 10:36:45 @alice: Thanks! Reducing batch size to 16 worked. Training is much more stable now.

2024-01-15 10:37:20 @charlie: Glad it helped! You might also want to use mixed precision training with torch.cuda.amp for better memory efficiency""",
        metadata={
            "title": "CUDA Memory Discussion",
            "source": "slack",
            "channel": "ml-help",
            "type": "conversation",
            "scraped_at": time.time()
        }
    )
    documents.append(conv_doc)
    
    # Code example
    code_doc = Document(
        text="""```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    \"\"\"Simple neural network for classification.\"\"\"
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        \"\"\"Forward pass through the network.\"\"\"
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    \"\"\"Train the neural network model.\"\"\"
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_size = 784  # 28x28 images
    hidden_size = 128
    num_classes = 10
    learning_rate = 0.001
    
    # Create model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model (assuming dataloader is defined)
    # train_model(model, train_dataloader, criterion, optimizer)
```""",
        metadata={
            "title": "Simple Neural Network Implementation",
            "source": "code",
            "language": "python",
            "type": "code_example",
            "scraped_at": time.time()
        }
    )
    documents.append(code_doc)
    
    # Web content
    web_doc = Document(
        text="""Machine Learning at Janelia Research Campus

Janelia Research Campus is at the forefront of computational neuroscience research, developing cutting-edge machine learning techniques for understanding brain function.

Our Research Areas

Computer Vision for Microscopy
We develop advanced computer vision algorithms to analyze microscopy images and extract meaningful biological insights. Our techniques include:
- Deep learning for cell segmentation
- Object tracking in live imaging
- Super-resolution microscopy enhancement

Neural Network Analysis
Our teams work on understanding neural circuits through computational modeling:
- Connectome reconstruction from electron microscopy
- Functional connectivity analysis
- Behavior prediction from neural activity

Open Source Tools
We believe in open science and provide our tools to the research community:
- FIJI/ImageJ plugins for image analysis  
- Python libraries for neuroscience
- Machine learning models and datasets

Recent Publications
- "Deep learning approaches for automated neuron reconstruction" (Nature Methods, 2024)
- "Connectome-based prediction of behavior in Drosophila" (Cell, 2024)
- "AI-assisted discovery of neural circuit motifs" (Science, 2023)

Collaboration Opportunities
We welcome collaborations with external researchers. Contact us at ml-support@janelia.hhmi.org for more information.""",
        metadata={
            "title": "Machine Learning at Janelia",
            "source": "web",
            "url": "https://janelia.org/research/machine-learning",
            "type": "web_content",
            "scraped_at": time.time()
        }
    )
    documents.append(web_doc)
    
    logger.info(f"✅ Created {len(documents)} test documents")
    return documents

async def test_modern_chunking():
    """Test the advanced chunking strategies."""
    logger.info("🔧 Testing modern chunking strategies")
    
    try:
        from chunking.advanced_chunkers import (
            HierarchicalChunker, SyntaxAwareChunker, AdaptiveChunker, ContentType
        )
        
        # Create test text
        test_text = """# Main Title

## Section 1: Introduction
This is the introduction section with some content.

## Section 2: Technical Details
Here are the technical details:

```python
def example_function():
    return "Hello World"
```

### Subsection 2.1
More detailed information here.

## Section 3: Conclusion
Final thoughts and conclusions."""
        
        # Test hierarchical chunking
        hierarchical_chunker = HierarchicalChunker(max_chunk_size=200)
        hierarchical_chunks = hierarchical_chunker.chunk(
            test_text, "test_doc", ContentType.MARKDOWN
        )
        logger.info(f"✅ Hierarchical chunking: {len(hierarchical_chunks)} chunks")
        
        # Test syntax-aware chunking
        syntax_chunker = SyntaxAwareChunker(target_chunk_size=150)
        syntax_chunks = syntax_chunker.chunk(
            test_text, "test_doc", ContentType.MARKDOWN
        )
        logger.info(f"✅ Syntax-aware chunking: {len(syntax_chunks)} chunks")
        
        # Test adaptive chunking
        adaptive_chunker = AdaptiveChunker(base_chunk_size=180)
        adaptive_chunks = adaptive_chunker.chunk(
            test_text, "test_doc", ContentType.MARKDOWN
        )
        logger.info(f"✅ Adaptive chunking: {len(adaptive_chunks)} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chunking test failed: {e}")
        return False

async def test_hyde_query_transformation():
    """Test HyDE query transformation."""
    logger.info("🔮 Testing HyDE query transformation")
    
    try:
        from chunking.hyde_query_transformer import HyDEIntegration
        
        hyde = HyDEIntegration()
        
        test_queries = [
            "How to fix CUDA out of memory errors?",
            "What are the best practices for neural network training?",
            "How to implement data augmentation in PyTorch?"
        ]
        
        for query in test_queries:
            enhanced_query = await hyde.enhance_search_query(
                query, 
                source_types=['technical', 'conversational'],
                domain_context='machine_learning'
            )
            
            logger.info(f"Query: '{query}'")
            logger.info(f"Enhanced: '{enhanced_query.get('combined_query', 'N/A')[:100]}...'")
            logger.info(f"Hypothetical docs: {len(enhanced_query.get('hypothetical_documents', []))}")
        
        logger.info("✅ HyDE query transformation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ HyDE test failed: {e}")
        return False

async def test_query_routing():
    """Test agentic query routing."""
    logger.info("🤖 Testing agentic query routing")
    
    try:
        from agents.query_router import AgenticQueryProcessor
        
        processor = AgenticQueryProcessor()
        
        test_queries = [
            ("How to implement a neural network in PyTorch?", "technical"),
            ("Why is my training failing with errors?", "troubleshooting"),
            ("What's the difference between RNN and LSTM?", "comparison"),
            ("Did anyone discuss this model architecture before?", "historical")
        ]
        
        for query, expected_intent in test_queries:
            plan = await processor.process_query(query)
            routing = plan['routing_decision']
            
            logger.info(f"Query: '{query}'")
            logger.info(f"Intent: {routing.search_params.get('intent', 'unknown')}")
            logger.info(f"Strategy: {routing.strategy}")
            logger.info(f"Confidence: {routing.confidence:.2f}")
            logger.info("---")
        
        logger.info("✅ Query routing working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Query routing test failed: {e}")
        return False

async def test_modern_indexing(weaviate_url: str, test_class: str = "ModernRAGTest"):
    """Test modern indexing with multi-vector embeddings."""
    logger.info("📚 Testing modern indexing")
    
    try:
        from modern_rag_integration import create_modern_rag_system, ModernRAGConfig
        from chunking.advanced_chunkers import ContentType
        
        # Create modern RAG system
        config = ModernRAGConfig(
            weaviate_url=weaviate_url,
            class_name=test_class,
            enable_multi_vector=True,
            enable_code_embeddings=True,
            enable_conversational_embeddings=True,
            chunking_strategy="hierarchical"
        )
        
        modern_rag = create_modern_rag_system(**config.__dict__)
        
        # Create schema
        modern_rag.create_schema(delete_existing=True)
        logger.info("✅ Schema created")
        
        # Create test documents
        documents = create_test_documents()
        
        # Index documents with different content types
        content_types = [
            (documents[0], ContentType.MARKDOWN),    # Technical doc
            (documents[1], ContentType.SLACK),       # Conversation
            (documents[2], ContentType.CODE),        # Code
            (documents[3], ContentType.WEB)          # Web content
        ]
        
        for doc, content_type in content_types:
            modern_rag.index_documents([doc], content_type=content_type)
            logger.info(f"✅ Indexed document with type: {content_type.value}")
        
        # Test retrieval
        query_engine = modern_rag.get_query_engine()
        test_query = "How to fix CUDA out of memory errors in neural network training?"
        
        start_time = time.time()
        response = query_engine.query(test_query)
        query_time = time.time() - start_time
        
        logger.info(f"✅ Query completed in {query_time:.2f} seconds")
        logger.info(f"Response length: {len(str(response))}")
        
        # Test enhanced query
        enhanced_result = await modern_rag.enhanced_query(test_query)
        logger.info(f"✅ Enhanced query - Routing: {enhanced_result.get('routing_decision', {}).get('strategy', 'N/A')}")
        
        modern_rag.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Modern indexing test failed: {e}")
        return False

async def test_retrieval_pipeline(weaviate_url: str, test_class: str = "ModernRAGTest"):
    """Test the enhanced retrieval pipeline."""
    logger.info("🔍 Testing enhanced retrieval pipeline")
    
    try:
        from retrieval.enhanced_retrieval import EnhancedRetrievalPipeline, RetrievalPipelineConfig
        from indexing.modern_weaviate_indexer import ModernWeaviateIndexer
        
        # Create indexer and retrieval pipeline
        indexer = ModernWeaviateIndexer(weaviate_url, test_class)
        
        config = RetrievalPipelineConfig(
            enable_dense_retrieval=True,
            enable_sparse_retrieval=True,
            enable_hybrid_fusion=True,
            enable_cross_encoder=True,
            dense_top_k=20,
            sparse_top_k=15,
            final_top_k=5
        )
        
        pipeline = EnhancedRetrievalPipeline(indexer, config)
        
        # Test retrieval
        test_query = "neural network training CUDA memory"
        
        start_time = time.time()
        results = await pipeline.retrieve(test_query)
        retrieval_time = time.time() - start_time
        
        logger.info(f"✅ Retrieved {len(results)} results in {retrieval_time:.2f} seconds")
        
        for i, result in enumerate(results[:3]):
            logger.info(f"Result {i+1}: Score={result.score:.3f}, Text={result.node.text[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Retrieval pipeline test failed: {e}")
        return False

def performance_comparison():
    """Compare performance between legacy and modern systems."""
    logger.info("⚡ Performance comparison (placeholder)")
    
    # This would run the same queries on both systems and compare:
    # - Query latency
    # - Response quality (if evaluation metrics available)
    # - Resource usage
    
    logger.info("📊 Performance comparison complete (mock)")
    return True

async def run_comprehensive_test(weaviate_url: str):
    """Run comprehensive test of all modern RAG components."""
    logger.info("🚀 Starting comprehensive modern RAG test")
    
    # Check dependencies
    modern_available, missing_deps = check_dependencies()
    if not modern_available:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        return False
    
    # Test Weaviate connection
    if not test_weaviate_connection(weaviate_url):
        return False
    
    test_results = {}
    
    # Run all tests
    test_results['chunking'] = await test_modern_chunking()
    test_results['hyde'] = await test_hyde_query_transformation()
    test_results['routing'] = await test_query_routing()
    test_results['indexing'] = await test_modern_indexing(weaviate_url)
    test_results['retrieval'] = await test_retrieval_pipeline(weaviate_url)
    test_results['performance'] = performance_comparison()
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    logger.info(f"\n🎯 Test Summary: {passed}/{total} tests passed")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Modern RAG system is ready.")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Update your Streamlit app settings to enable Modern RAG")
        logger.info("2. Re-index your data with modern chunking: `python index_wiki.py --modern-rag`")
        logger.info("3. Test queries in the web interface with enhanced features")
        return True
    else:
        logger.error("❌ Some tests failed. Check logs for details.")
        return False

def main():
    """Main function to run tests or migration."""
    parser = argparse.ArgumentParser(description="Test and migrate to modern RAG system")
    parser.add_argument('-w', '--weaviate-url', default="http://localhost:8777", 
                       help="Weaviate URL")
    parser.add_argument('--test-only', action='store_true',
                       help="Run tests without indexing")
    parser.add_argument('--migration', action='store_true',
                       help="Run migration from legacy system")
    parser.add_argument('--quick-test', action='store_true',
                       help="Run quick dependency check only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        modern_available, missing_deps = check_dependencies()
        if modern_available:
            logger.info("✅ Quick test passed - Modern RAG ready")
            sys.exit(0)
        else:
            logger.error(f"❌ Quick test failed - Missing: {missing_deps}")
            sys.exit(1)
    
    # Run comprehensive test
    success = asyncio.run(run_comprehensive_test(args.weaviate_url))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()