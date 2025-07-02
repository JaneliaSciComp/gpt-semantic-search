#!/usr/bin/env python3
"""Quick verification of modern RAG availability"""

def verify_imports():
    """Verify all modern RAG components can be imported"""
    try:
        from modern_rag_integration import create_modern_rag_system, ModernRAGConfig
        print("✅ Modern RAG integration")
        
        from chunking.advanced_chunkers import HierarchicalChunker, SyntaxAwareChunker, AdaptiveChunker
        print("✅ Advanced chunkers")
        
        from chunking.hyde_query_transformer import HyDEIntegration
        print("✅ HyDE query transformer")
        
        from agents.query_router import AgenticQueryProcessor
        print("✅ Agentic query processor")
        
        from indexing.modern_weaviate_indexer import ModernWeaviateIndexer
        print("✅ Modern Weaviate indexer")
        
        from retrieval.enhanced_retrieval import EnhancedRetrievalPipeline
        print("✅ Enhanced retrieval pipeline")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def verify_weaviate_connection():
    """Verify Weaviate connection"""
    try:
        import weaviate
        client = weaviate.Client("http://localhost:8777")
        if client.is_live() and client.is_ready():
            print("✅ Weaviate connection")
            return True
        else:
            print("❌ Weaviate not ready")
            return False
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        return False

def verify_openai_access():
    """Verify OpenAI API access"""
    try:
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        print("✅ OpenAI LLM initialized")
        
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed = OpenAIEmbedding(model="text-embedding-3-large")
        print("✅ OpenAI embeddings initialized")
        
        return True
    except Exception as e:
        print(f"❌ OpenAI access failed: {e}")
        return False

def main():
    print("🔍 Verifying Modern RAG System...")
    
    results = []
    results.append(verify_imports())
    results.append(verify_weaviate_connection())
    results.append(verify_openai_access())
    
    if all(results):
        print("\n🎉 Modern RAG System Ready!")
        print("\nNext steps:")
        print("1. Run: pixi run streamlit run 1_🔍_Search.py")
        print("2. Go to Settings and enable Modern RAG features")
        print("3. Test queries with enhanced capabilities")
        return True
    else:
        print("\n❌ Some components not ready")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)