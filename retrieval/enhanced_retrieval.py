"""
Enhanced Multi-Stage Retrieval Pipeline with Reranking

Implements advanced retrieval strategies combining:
- Dense + Sparse + Cross-encoder pipeline
- Query expansion and transformation
- Contextual retrieval with hierarchical context
- Multi-vector fusion and reranking
- Temporal and relevance filtering
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from abc import ABC, abstractmethod

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import QueryBundle

from chunking.hyde_query_transformer import HyDEIntegration
from agents.query_router import AgenticQueryProcessor, RoutingDecision
from indexing.modern_weaviate_indexer import ModernWeaviateIndexer

logger = logging.getLogger(__name__)

class RetrievalStage(Enum):
    """Stages in the retrieval pipeline."""
    QUERY_PROCESSING = "query_processing"
    DENSE_RETRIEVAL = "dense_retrieval"
    SPARSE_RETRIEVAL = "sparse_retrieval"
    HYBRID_FUSION = "hybrid_fusion"
    CROSS_ENCODER_RERANK = "cross_encoder_rerank"
    CONTEXTUAL_EXPANSION = "contextual_expansion"
    FINAL_RANKING = "final_ranking"

@dataclass
class RetrievalResult:
    """Result from a retrieval stage."""
    node_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    source_stage: RetrievalStage
    embedding_type: Optional[str] = None
    
    def to_node_with_score(self) -> NodeWithScore:
        """Convert to LlamaIndex NodeWithScore."""
        from llama_index.core.schema import TextNode
        
        node = TextNode(
            text=self.content,
            id_=self.node_id,
            metadata=self.metadata
        )
        return NodeWithScore(node=node, score=self.score)

@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    # Stage configurations
    enable_dense_retrieval: bool = True
    enable_sparse_retrieval: bool = True
    enable_hybrid_fusion: bool = True
    enable_cross_encoder: bool = True
    enable_contextual_expansion: bool = True
    
    # Retrieval parameters
    dense_top_k: int = 50
    sparse_top_k: int = 30
    hybrid_top_k: int = 20
    final_top_k: int = 10
    
    # Fusion parameters
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    cross_encoder_weight: float = 0.8
    contextual_weight: float = 0.2
    
    # Cross-encoder configuration
    cross_encoder_model: str = "ms-marco-MiniLM-L-6-v2"
    cross_encoder_threshold: float = 0.5
    
    # Query expansion
    enable_query_expansion: bool = True
    expansion_terms: int = 5
    
    # Temporal filtering
    enable_temporal_boost: bool = True
    recency_decay_days: int = 365
    temporal_boost_factor: float = 1.2

class BaseRetrievalStage(ABC):
    """Abstract base class for retrieval stages."""
    
    @abstractmethod
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Process the retrieval stage."""
        pass

class DenseRetrievalStage(BaseRetrievalStage):
    """Dense vector retrieval stage."""
    
    def __init__(self, 
                 indexer: ModernWeaviateIndexer,
                 embedding_model: OpenAIEmbedding,
                 top_k: int = 50):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Perform dense vector retrieval."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.get_text_embedding(query)
            
            # Search using dense vectors
            results = self.indexer.search_multi_vector(
                query=query,
                vector_names=["dense"],
                limit=self.top_k,
                content_type=context.get('content_type')
            )
            
            # Convert to RetrievalResult
            retrieval_results = []
            for result in results:
                retrieval_result = RetrievalResult(
                    node_id=result["uuid"],
                    score=result["score"],
                    content=result["properties"]["text"],
                    metadata=result["properties"],
                    source_stage=RetrievalStage.DENSE_RETRIEVAL,
                    embedding_type="dense"
                )
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Dense retrieval returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

class SparseRetrievalStage(BaseRetrievalStage):
    """Sparse/keyword retrieval stage using BM25."""
    
    def __init__(self, 
                 indexer: ModernWeaviateIndexer,
                 top_k: int = 30):
        self.indexer = indexer
        self.top_k = top_k
    
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Perform sparse/BM25 retrieval using v3 API."""
        try:
            # Build query with BM25 search
            query_builder = self.indexer.client.query.get(self.indexer.class_name, [
                "ref_doc_id", "text", "title", "link", "source", "content_type",
                "chunk_id", "parent_id", "level", "token_count"
            ]).with_bm25(query=query).with_limit(self.top_k)
            
            # Add content type filter if specified
            if context.get('content_type'):
                query_builder = query_builder.with_where({
                    "path": ["content_type"],
                    "operator": "Equal",
                    "valueText": context['content_type']
                })
            
            # Add additional metadata
            query_builder = query_builder.with_additional(["score"])
            
            # Execute query
            result = query_builder.do()
            
            # Convert to RetrievalResult
            retrieval_results = []
            if "data" in result and "Get" in result["data"] and self.indexer.class_name in result["data"]["Get"]:
                for obj in result["data"]["Get"][self.indexer.class_name]:
                    additional = obj.get("_additional", {})
                    retrieval_result = RetrievalResult(
                        node_id=additional.get("id", ""),
                        score=additional.get("score", 0.0),
                        content=obj.get("text", ""),
                        metadata={k: v for k, v in obj.items() if not k.startswith("_")},
                        source_stage=RetrievalStage.SPARSE_RETRIEVAL
                    )
                    retrieval_results.append(retrieval_result)
            
            logger.info(f"Sparse retrieval returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []

class HybridFusionStage(BaseRetrievalStage):
    """Fusion stage combining dense and sparse retrieval results."""
    
    def __init__(self, 
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 top_k: int = 20):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
    
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion."""
        try:
            # Separate dense and sparse results
            dense_results = [r for r in previous_results if r.source_stage == RetrievalStage.DENSE_RETRIEVAL]
            sparse_results = [r for r in previous_results if r.source_stage == RetrievalStage.SPARSE_RETRIEVAL]
            
            # Create rank mappings
            dense_ranks = {result.node_id: i + 1 for i, result in enumerate(dense_results)}
            sparse_ranks = {result.node_id: i + 1 for i, result in enumerate(sparse_results)}
            
            # Collect all unique node IDs
            all_node_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
            
            # Calculate RRF scores
            fused_results = []
            k = 60  # RRF parameter
            
            for node_id in all_node_ids:
                # Get original results
                dense_result = next((r for r in dense_results if r.node_id == node_id), None)
                sparse_result = next((r for r in sparse_results if r.node_id == node_id), None)
                
                # Calculate RRF score
                dense_rrf = self.dense_weight / (k + dense_ranks.get(node_id, len(dense_results) + 1))
                sparse_rrf = self.sparse_weight / (k + sparse_ranks.get(node_id, len(sparse_results) + 1))
                combined_score = dense_rrf + sparse_rrf
                
                # Use the result with content (prefer dense if both exist)
                base_result = dense_result or sparse_result
                if base_result:
                    fused_result = RetrievalResult(
                        node_id=node_id,
                        score=combined_score,
                        content=base_result.content,
                        metadata=base_result.metadata,
                        source_stage=RetrievalStage.HYBRID_FUSION
                    )
                    fused_results.append(fused_result)
            
            # Sort by combined score and take top_k
            fused_results.sort(key=lambda x: x.score, reverse=True)
            final_results = fused_results[:self.top_k]
            
            logger.info(f"Hybrid fusion produced {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid fusion failed: {e}")
            return previous_results[:self.top_k]

class CrossEncoderReranking(BaseRetrievalStage):
    """Cross-encoder reranking stage for improved relevance."""
    
    def __init__(self, 
                 llm: OpenAI,
                 top_k: int = 10,
                 threshold: float = 0.5):
        self.llm = llm
        self.top_k = top_k
        self.threshold = threshold
    
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder scoring."""
        try:
            if not previous_results:
                return []
            
            # Score each result with LLM
            scored_results = []
            for result in previous_results:
                relevance_score = await self._score_relevance(query, result.content)
                
                # Combine with previous score
                combined_score = 0.7 * relevance_score + 0.3 * result.score
                
                if relevance_score >= self.threshold:
                    reranked_result = RetrievalResult(
                        node_id=result.node_id,
                        score=combined_score,
                        content=result.content,
                        metadata=result.metadata,
                        source_stage=RetrievalStage.CROSS_ENCODER_RERANK
                    )
                    scored_results.append(reranked_result)
            
            # Sort by combined score
            scored_results.sort(key=lambda x: x.score, reverse=True)
            final_results = scored_results[:self.top_k]
            
            logger.info(f"Cross-encoder reranking produced {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return previous_results[:self.top_k]
    
    async def _score_relevance(self, query: str, content: str) -> float:
        """Score relevance using LLM."""
        prompt = f"""
        Rate the relevance of this content to the query on a scale of 0.0 to 1.0.
        
        Query: "{query}"
        
        Content: "{content[:1000]}..."
        
        Consider:
        - How directly the content answers the query
        - Accuracy and completeness of information
        - Specificity and usefulness
        
        Respond with only a number between 0.0 and 1.0:
        """
        
        try:
            response = await self.llm.acomplete(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM relevance scoring failed: {e}")
            return 0.5

class ContextualExpansionStage(BaseRetrievalStage):
    """Contextual expansion using hierarchical document relationships."""
    
    def __init__(self, 
                 indexer: ModernWeaviateIndexer,
                 expansion_factor: float = 0.3):
        self.indexer = indexer
        self.expansion_factor = expansion_factor
    
    async def process(self, 
                     query: str,
                     previous_results: List[RetrievalResult],
                     context: Dict[str, Any]) -> List[RetrievalResult]:
        """Expand results with hierarchical context."""
        try:
            expanded_results = list(previous_results)
            
            for result in previous_results:
                # Get hierarchical context
                hierarchical_context = self.indexer.get_hierarchical_context(
                    result.node_id, include_children=True
                )
                
                # Add parent context if available
                if hierarchical_context.get('parent'):
                    parent_content = hierarchical_context['parent']['text']
                    parent_result = RetrievalResult(
                        node_id=hierarchical_context['parent']['chunk_id'],
                        score=result.score * self.expansion_factor,
                        content=parent_content,
                        metadata=hierarchical_context['parent'],
                        source_stage=RetrievalStage.CONTEXTUAL_EXPANSION
                    )
                    expanded_results.append(parent_result)
                
                # Add relevant children
                for child in hierarchical_context.get('children', [])[:2]:  # Limit to 2 children
                    child_result = RetrievalResult(
                        node_id=child['chunk_id'],
                        score=result.score * self.expansion_factor * 0.8,
                        content=child['text'],
                        metadata=child,
                        source_stage=RetrievalStage.CONTEXTUAL_EXPANSION
                    )
                    expanded_results.append(child_result)
            
            # Remove duplicates and sort
            unique_results = {r.node_id: r for r in expanded_results}.values()
            final_results = sorted(unique_results, key=lambda x: x.score, reverse=True)
            
            logger.info(f"Contextual expansion produced {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Contextual expansion failed: {e}")
            return previous_results

class EnhancedRetrievalPipeline:
    """
    Complete enhanced retrieval pipeline orchestrating all stages.
    """
    
    def __init__(self, 
                 indexer: ModernWeaviateIndexer,
                 config: Optional[RetrievalPipelineConfig] = None,
                 llm_model: str = "gpt-4o-mini"):
        self.indexer = indexer
        self.config = config or RetrievalPipelineConfig()
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.embedding_model = OpenAIEmbedding(model="text-embedding-3-large")
        
        # Initialize query processor and HyDE
        self.query_processor = AgenticQueryProcessor(llm_model)
        self.hyde_integration = HyDEIntegration(llm_model)
        
        # Initialize retrieval stages
        self.stages = {}
        if self.config.enable_dense_retrieval:
            self.stages[RetrievalStage.DENSE_RETRIEVAL] = DenseRetrievalStage(
                self.indexer, self.embedding_model, self.config.dense_top_k
            )
        
        if self.config.enable_sparse_retrieval:
            self.stages[RetrievalStage.SPARSE_RETRIEVAL] = SparseRetrievalStage(
                self.indexer, self.config.sparse_top_k
            )
        
        if self.config.enable_hybrid_fusion:
            self.stages[RetrievalStage.HYBRID_FUSION] = HybridFusionStage(
                self.config.dense_weight, self.config.sparse_weight, self.config.hybrid_top_k
            )
        
        if self.config.enable_cross_encoder:
            self.stages[RetrievalStage.CROSS_ENCODER_RERANK] = CrossEncoderReranking(
                self.llm, self.config.final_top_k, self.config.cross_encoder_threshold
            )
        
        if self.config.enable_contextual_expansion:
            self.stages[RetrievalStage.CONTEXTUAL_EXPANSION] = ContextualExpansionStage(
                self.indexer, self.config.contextual_weight
            )
    
    async def retrieve(self, 
                      query: str,
                      context: Optional[Dict[str, Any]] = None) -> List[NodeWithScore]:
        """
        Execute the complete enhanced retrieval pipeline.
        
        Args:
            query: User query
            context: Additional context for retrieval
            
        Returns:
            List of NodeWithScore objects
        """
        context = context or {}
        
        # Step 1: Process query with agentic router
        processing_plan = await self.query_processor.process_query(query, context)
        routing_decision = processing_plan['routing_decision']
        
        # Step 2: Apply HyDE query transformation if beneficial
        enhanced_query_components = await self._apply_hyde_if_beneficial(
            query, routing_decision, context
        )
        
        # Step 3: Execute retrieval pipeline stages
        results = []
        current_results = []
        
        # Dense retrieval
        if RetrievalStage.DENSE_RETRIEVAL in self.stages:
            search_query = enhanced_query_components.get('combined_query', query)
            dense_results = await self.stages[RetrievalStage.DENSE_RETRIEVAL].process(
                search_query, current_results, context
            )
            results.extend(dense_results)
            current_results.extend(dense_results)
        
        # Sparse retrieval  
        if RetrievalStage.SPARSE_RETRIEVAL in self.stages:
            sparse_results = await self.stages[RetrievalStage.SPARSE_RETRIEVAL].process(
                query, current_results, context
            )
            results.extend(sparse_results)
            current_results.extend(sparse_results)
        
        # Hybrid fusion
        if RetrievalStage.HYBRID_FUSION in self.stages:
            fused_results = await self.stages[RetrievalStage.HYBRID_FUSION].process(
                query, current_results, context
            )
            current_results = fused_results
        
        # Cross-encoder reranking
        if RetrievalStage.CROSS_ENCODER_RERANK in self.stages:
            reranked_results = await self.stages[RetrievalStage.CROSS_ENCODER_RERANK].process(
                query, current_results, context
            )
            current_results = reranked_results
        
        # Contextual expansion
        if RetrievalStage.CONTEXTUAL_EXPANSION in self.stages:
            expanded_results = await self.stages[RetrievalStage.CONTEXTUAL_EXPANSION].process(
                query, current_results, context
            )
            current_results = expanded_results
        
        # Apply temporal boosting
        if self.config.enable_temporal_boost:
            current_results = self._apply_temporal_boosting(current_results)
        
        # Convert to NodeWithScore and return top results
        final_results = [result.to_node_with_score() for result in current_results[:self.config.final_top_k]]
        
        logger.info(f"Enhanced retrieval pipeline returned {len(final_results)} final results")
        return final_results
    
    async def _apply_hyde_if_beneficial(self, 
                                      query: str,
                                      routing_decision: RoutingDecision,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply HyDE if it would be beneficial for this query type."""
        # Determine if HyDE would help based on routing decision
        use_hyde = (
            routing_decision.strategy in ['technical_deep_dive', 'comparative_analysis'] or
            routing_decision.confidence < 0.7
        )
        
        if use_hyde:
            # Determine content types for HyDE
            source_types = []
            for source in routing_decision.data_sources:
                if source.value == 'slack':
                    source_types.append('conversational')
                elif source.value == 'wiki':
                    source_types.append('wiki')
                else:
                    source_types.append('technical')
            
            # Apply HyDE
            enhanced_components = await self.hyde_integration.enhance_search_query(
                query, source_types, context.get('domain', 'janelia')
            )
            return enhanced_components
        else:
            return {'combined_query': query}
    
    def _apply_temporal_boosting(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply temporal boosting to favor recent content."""
        import time
        current_time = time.time()
        
        for result in results:
            scraped_at = result.metadata.get('scraped_at', current_time)
            days_old = (current_time - scraped_at) / (24 * 3600)
            
            if days_old < self.config.recency_decay_days:
                # Apply temporal boost (newer content gets higher scores)
                temporal_factor = 1.0 - (days_old / self.config.recency_decay_days)
                boost = 1.0 + (temporal_factor * (self.config.temporal_boost_factor - 1.0))
                result.score *= boost
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class ModernRAGRetriever(BaseRetriever):
    """
    LlamaIndex-compatible retriever using the enhanced pipeline.
    """
    
    def __init__(self, 
                 pipeline: EnhancedRetrievalPipeline,
                 **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using enhanced pipeline."""
        # Convert to async call
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.pipeline.retrieve(query_bundle.query_str))
                return future.result()
        else:
            return asyncio.run(self.pipeline.retrieve(query_bundle.query_str))
    
    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes using enhanced pipeline."""
        return await self.pipeline.retrieve(query_bundle.query_str)