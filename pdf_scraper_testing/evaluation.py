import logging
import weaviate
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import existing evaluation framework
import sys
sys.path.append('..')
from indexing.weaviate_indexer import Indexer, _class_name

# Import existing evaluation service
sys.path.append('../eval/RAGASEval')
from generateAnswer import SemanticSearchService

logger = logging.getLogger(__name__)


class PDFScraperEvaluator:
    """Evaluator for comparing PDF scrapers using Weaviate vector search"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080", test_class_prefix: str = "PDFTest"):
        self.weaviate_url = weaviate_url
        self.test_class_prefix = test_class_prefix
        self.client = weaviate.Client(weaviate_url)
        
        if not self.client.is_live():
            raise ConnectionError(f"Weaviate is not live at {weaviate_url}")
    
    def get_available_scrapers(self) -> List[str]:
        """Get list of scrapers that have been indexed"""
        schema = self.client.schema.get()
        classes = schema["classes"]
        
        scrapers = []
        for class_info in classes:
            class_name = class_info["class"]
            if class_name.startswith(self.test_class_prefix):
                scraper_name = class_name.replace(f"{self.test_class_prefix}_", "").replace("_Node", "")
                scrapers.append(scraper_name)
        
        return scrapers
    
    def search_with_scraper(self, query: str, scraper_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search using a specific scraper's indexed content"""
        class_name = _class_name(f"{self.test_class_prefix}_{scraper_name}")
        
        try:
            result = (
                self.client.query
                .get(class_name, ["text", "title", "source", "link", "scraped_at", "scraper_used", "processing_time", "file_size", "characters_extracted"])
                .with_hybrid(query=query)
                .with_limit(limit)
                .do()
            )
            
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"][class_name]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Search failed for {scraper_name}: {e}")
            return []
    
    def create_search_service(self, scraper_name: str) -> SemanticSearchService:
        """Create a search service for a specific scraper"""
        class CustomSearchService(SemanticSearchService):
            def __init__(self, weaviate_url, class_prefix):
                self.weaviate_url = weaviate_url
                self.class_prefix = class_prefix
                self.weaviate_client = self.get_weaviate_client()
                self.query_engine = self.get_query_engine()
            
            def get_query_engine(self):
                from llama_index.llms.openai import OpenAI
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.core import Settings, PromptHelper, GPTVectorStoreIndex, StorageContext
                from llama_index.core.retrievers import VectorIndexRetriever
                from llama_index.core.query_engine import RetrieverQueryEngine
                from llama_index.vector_stores.weaviate import WeaviateVectorStore
                from llama_index.core.vector_stores.types import VectorStoreQueryMode
                
                llm = OpenAI(model="gpt-4o-mini", temperature=0)
                embed_model = OpenAIEmbedding(model="text-embedding-3-large")
                prompt_helper = PromptHelper(4096, 256, 0.1)

                Settings.llm = llm
                Settings.embed_model = embed_model
                Settings.chunk_size = 512
                Settings.prompt_helper = prompt_helper

                vector_store = WeaviateVectorStore(weaviate_client=self.weaviate_client, class_prefix=self.class_prefix)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = GPTVectorStoreIndex([], storage_context=storage_context)

                retriever = VectorIndexRetriever(
                    index,
                    similarity_top_k=5,
                    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                    alpha=0.5,
                )

                query_engine = RetrieverQueryEngine.from_args(retriever)
                return query_engine
        
        return CustomSearchService(self.weaviate_url, f"{self.test_class_prefix}_{scraper_name}")
    
    def compare_search_results(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Compare search results across all available scrapers"""
        scrapers = self.get_available_scrapers()
        
        if not scrapers:
            logger.warning("No scrapers found in Weaviate")
            return {}
        
        comparison = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'scrapers': {}
        }
        
        for scraper in scrapers:
            logger.info(f"Searching with {scraper}")
            results = self.search_with_scraper(query, scraper, limit)
            
            comparison['scrapers'][scraper] = {
                'results_count': len(results),
                'results': results
            }
        
        return comparison
    
    def evaluate_retrieval_quality(self, test_queries: List[str], limit: int = 10) -> pd.DataFrame:
        """Evaluate retrieval quality across scrapers for multiple queries"""
        evaluation_data = []
        
        for query in test_queries:
            logger.info(f"Evaluating query: {query}")
            comparison = self.compare_search_results(query, limit)
            
            for scraper, scraper_results in comparison['scrapers'].items():
                # Calculate basic metrics
                results_count = scraper_results['results_count']
                
                # Calculate average relevance score (if available)
                avg_score = 0
                if scraper_results['results']:
                    scores = []
                    for result in scraper_results['results']:
                        if '_additional' in result and 'score' in result['_additional']:
                            scores.append(result['_additional']['score'])
                    avg_score = sum(scores) / len(scores) if scores else 0
                
                evaluation_data.append({
                    'query': query,
                    'scraper': scraper,
                    'results_count': results_count,
                    'avg_relevance_score': avg_score
                })
        
        return pd.DataFrame(evaluation_data)
    
    def analyze_content_coverage(self) -> pd.DataFrame:
        """Analyze how much content each scraper extracted"""
        scrapers = self.get_available_scrapers()
        coverage_data = []
        
        for scraper in scrapers:
            class_name = _class_name(f"{self.test_class_prefix}_{scraper}")
            
            try:
                # Get aggregated statistics
                result = (
                    self.client.query
                    .aggregate(class_name)
                    .with_fields("characters_extracted { count sum mean maximum minimum }")
                    .with_fields("file_size { count sum mean maximum minimum }")
                    .with_fields("processing_time { count sum mean maximum minimum }")
                    .do()
                )
                
                if "data" in result and "Aggregate" in result["data"]:
                    stats = result["data"]["Aggregate"][class_name][0]
                    
                    coverage_data.append({
                        'scraper': scraper,
                        'document_count': stats.get('characters_extracted', {}).get('count', 0),
                        'total_characters': stats.get('characters_extracted', {}).get('sum', 0),
                        'avg_characters': stats.get('characters_extracted', {}).get('mean', 0),
                        'max_characters': stats.get('characters_extracted', {}).get('maximum', 0),
                        'min_characters': stats.get('characters_extracted', {}).get('minimum', 0),
                        'avg_file_size': stats.get('file_size', {}).get('mean', 0),
                        'avg_processing_time': stats.get('processing_time', {}).get('mean', 0),
                        'total_processing_time': stats.get('processing_time', {}).get('sum', 0)
                    })
            
            except Exception as e:
                logger.error(f"Error analyzing coverage for {scraper}: {e}")
        
        return pd.DataFrame(coverage_data)
    
    def generate_evaluation_report(self, test_queries: List[str]) -> str:
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report")
        
        # Get basic info
        scrapers = self.get_available_scrapers()
        
        if not scrapers:
            return "No scrapers found in Weaviate for evaluation"
        
        # Analyze content coverage
        coverage_df = self.analyze_content_coverage()
        
        # Evaluate retrieval quality
        retrieval_df = self.evaluate_retrieval_quality(test_queries)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# PDF Scraper Evaluation Report
Generated: {timestamp}

## Available Scrapers
{', '.join(scrapers)}

## Content Coverage Analysis
{coverage_df.to_string(index=False)}

## Retrieval Quality Analysis
{retrieval_df.to_string(index=False)}

## Summary Statistics

### By Scraper Performance
{retrieval_df.groupby('scraper').agg({
    'results_count': ['mean', 'std'],
    'avg_relevance_score': ['mean', 'std']
}).round(3).to_string()}

### By Query Performance
{retrieval_df.groupby('query').agg({
    'results_count': ['mean', 'std'],
    'avg_relevance_score': ['mean', 'std']
}).round(3).to_string()}

## Recommendations

Based on the analysis:
"""
        
        # Add recommendations based on data
        if not coverage_df.empty:
            best_coverage = coverage_df.loc[coverage_df['avg_characters'].idxmax()]
            fastest_scraper = coverage_df.loc[coverage_df['avg_processing_time'].idxmin()]
            
            report += f"""
- **Best Content Coverage**: {best_coverage['scraper']} (avg {best_coverage['avg_characters']:.0f} characters)
- **Fastest Processing**: {fastest_scraper['scraper']} (avg {fastest_scraper['avg_processing_time']:.3f} seconds)
"""
        
        if not retrieval_df.empty:
            best_retrieval = retrieval_df.groupby('scraper')['avg_relevance_score'].mean().idxmax()
            most_results = retrieval_df.groupby('scraper')['results_count'].mean().idxmax()
            
            report += f"""
- **Best Retrieval Quality**: {best_retrieval}
- **Most Results Retrieved**: {most_results}
"""
        
        # Save report
        report_filename = f"pdf_scraper_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = Path("results") / report_filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report
    
    def cleanup_test_data(self):
        """Clean up test data from Weaviate"""
        schema = self.client.schema.get()
        classes = schema["classes"]
        
        classes_to_delete = []
        for class_info in classes:
            class_name = class_info["class"]
            if class_name.startswith(self.test_class_prefix):
                classes_to_delete.append(class_name)
        
        for class_name in classes_to_delete:
            logger.info(f"Deleting test class: {class_name}")
            self.client.schema.delete_class(class_name)
        
        logger.info("Test data cleanup completed")