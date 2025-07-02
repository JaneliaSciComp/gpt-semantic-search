#!/usr/bin/env python3
"""Quick test of modern RAG components"""

import asyncio
from modern_rag_integration import create_modern_rag_system, ModernRAGConfig
from chunking.hyde_query_transformer import HyDEIntegration
from agents.query_router import AgenticQueryProcessor

print('✅ All imports successful')

async def test_hyde():
    try:
        hyde = HyDEIntegration('gpt-4o-mini')
        result = await hyde.enhance_search_query('test query', ['technical'], 'test')
        docs_count = len(result.get("hypothetical_documents", []))
        print(f'✅ HyDE working: {docs_count} docs generated')
        return True
    except Exception as e:
        print(f'❌ HyDE failed: {e}')
        return False

async def test_routing():
    try:
        processor = AgenticQueryProcessor('gpt-4o-mini')
        plan = await processor.process_query('How to fix errors?')
        strategy = plan["routing_decision"].strategy
        print(f'✅ Query routing working: strategy={strategy}')
        return True
    except Exception as e:
        print(f'❌ Query routing failed: {e}')
        return False

async def main():
    results = []
    results.append(await test_hyde())
    results.append(await test_routing())
    
    if all(results):
        print('🎉 All modern RAG components working!')
        return True
    else:
        print('❌ Some components failed')
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)