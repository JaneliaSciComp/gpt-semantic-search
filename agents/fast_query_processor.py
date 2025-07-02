"""
Fast Query Processor for Real-time Performance

Optimized agentic query processing with configurable speed vs quality tradeoffs.
Includes streaming capabilities for intermediate step display.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum

from llama_index.llms.openai import OpenAI

from .query_router import AgenticQueryProcessor, QueryClassification, RoutingDecision

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing speed modes with different quality tradeoffs."""
    FAST = "fast"           # ~0.5-1s, basic routing, no complex analysis
    BALANCED = "balanced"   # ~1-3s, standard processing with optimizations  
    COMPREHENSIVE = "comprehensive"  # ~3-8s, full analysis and routing

@dataclass
class StreamingStep:
    """Represents a single step in the streaming process."""
    step_type: str
    step_name: str
    status: str  # "starting", "in_progress", "completed", "skipped", "error"
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class FastQueryProcessor:
    """
    High-performance query processor with streaming capabilities.
    
    Optimized for speed while maintaining quality through intelligent shortcuts
    and parallel processing where possible.
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                 enable_streaming: bool = True,
                 enable_parallel: bool = True):
        self.llm_model = llm_model
        self.processing_mode = processing_mode
        self.enable_streaming = enable_streaming
        self.enable_parallel = enable_parallel
        
        # Initialize components based on mode
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.base_processor = AgenticQueryProcessor(llm_model)
        
        # Performance configurations per mode
        self.mode_configs = {
            ProcessingMode.FAST: {
                "enable_pattern_shortcuts": True,
                "enable_llm_classification": False,
                "max_routing_time": 0.5,
                "skip_intent_analysis": True,
                "use_cached_patterns": True
            },
            ProcessingMode.BALANCED: {
                "enable_pattern_shortcuts": True,
                "enable_llm_classification": True,
                "max_routing_time": 2.0,
                "skip_intent_analysis": False,
                "use_cached_patterns": True
            },
            ProcessingMode.COMPREHENSIVE: {
                "enable_pattern_shortcuts": False,
                "enable_llm_classification": True,
                "max_routing_time": 5.0,
                "skip_intent_analysis": False,
                "use_cached_patterns": False
            }
        }
        
        self.config = self.mode_configs[processing_mode]
        
        # Pre-compiled patterns for fast mode
        self.quick_patterns = {
            "technical": ["how to", "implement", "configure", "setup", "install", "error", "fix", "debug"],
            "comparison": ["difference", "vs", "versus", "compare", "better", "which"],
            "troubleshooting": ["error", "problem", "issue", "fail", "not working", "broken"],
            "historical": ["who", "when", "did anyone", "previous", "before", "discussed"]
        }
    
    async def process_query_streaming(self, 
                                    query: str, 
                                    context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamingStep, None]:
        """
        Process query with streaming intermediate steps.
        
        Yields StreamingStep objects that can be consumed by the frontend
        for real-time display of processing steps.
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Step 1: Query Analysis
            yield StreamingStep("analysis", "Query Analysis", "starting", "Analyzing query intent...")
            
            classification_start = time.time()
            if self.config["enable_pattern_shortcuts"]:
                classification = await self._fast_classify_query(query)
                classification_time = time.time() - classification_start
                yield StreamingStep(
                    "analysis", "Quick Classification", "completed", 
                    f"Classified as '{classification.primary_intent}' using pattern matching",
                    {"intent": classification.primary_intent, "confidence": classification.confidence},
                    duration=classification_time
                )
            else:
                classification = await self._comprehensive_classify_query(query)
                classification_time = time.time() - classification_start
                yield StreamingStep(
                    "analysis", "LLM Classification", "completed",
                    f"Deep analysis completed: {classification.primary_intent}",
                    {"intent": classification.primary_intent, "confidence": classification.confidence},
                    duration=classification_time
                )
            
            # Step 2: Route Planning
            yield StreamingStep("routing", "Route Planning", "starting", "Determining optimal search strategy...")
            
            routing_start = time.time()
            routing_decision = await self._create_routing_decision(query, classification, context)
            routing_time = time.time() - routing_start
            
            yield StreamingStep(
                "routing", "Strategy Selection", "completed",
                f"Selected '{routing_decision.strategy}' strategy with {len(routing_decision.data_sources)} sources",
                {
                    "strategy": routing_decision.strategy,
                    "sources": [s.value for s in routing_decision.data_sources],
                    "confidence": routing_decision.confidence
                },
                duration=routing_time
            )
            
            # Step 3: Query Enhancement (if enabled)
            if self._should_use_query_enhancement(classification, routing_decision):
                yield StreamingStep("enhancement", "Query Enhancement", "starting", "Enhancing query for better retrieval...")
                
                enhancement_start = time.time()
                enhanced_query = await self._fast_enhance_query(query, classification)
                enhancement_time = time.time() - enhancement_start
                
                yield StreamingStep(
                    "enhancement", "Query Optimization", "completed",
                    f"Query enhanced with {len(enhanced_query.get('expansion_terms', []))} additional terms",
                    {"original_length": len(query), "enhanced_length": len(enhanced_query.get('enhanced_query', query))},
                    duration=enhancement_time
                )
            else:
                yield StreamingStep("enhancement", "Query Enhancement", "skipped", "Enhancement not needed for this query type")
            
            # Step 4: Finalization
            total_time = time.time() - start_time
            yield StreamingStep(
                "completion", "Processing Complete", "completed",
                f"Query processing completed in {total_time:.2f}s",
                {
                    "total_time": total_time,
                    "mode": self.processing_mode.value,
                    "final_routing": routing_decision.__dict__
                },
                duration=total_time
            )
            
        except Exception as e:
            yield StreamingStep(
                "error", "Processing Error", "error",
                f"Error during processing: {str(e)}",
                {"error_type": type(e).__name__, "error_message": str(e)}
            )
    
    async def process_query_fast(self, 
                               query: str, 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fast, non-streaming query processing for when speed is critical.
        """
        start_time = time.time()
        
        # Fast path with minimal processing
        if self.processing_mode == ProcessingMode.FAST:
            classification = await self._fast_classify_query(query)
            routing_decision = await self._fast_route_query(query, classification)
        else:
            # Use streaming but don't yield steps
            steps = []
            async for step in self.process_query_streaming(query, context):
                if step.step_type == "completion":
                    routing_decision = step.data["final_routing"]
                    break
        
        processing_time = time.time() - start_time
        
        return {
            "routing_decision": routing_decision,
            "processing_time": processing_time,
            "mode": self.processing_mode.value
        }
    
    async def _fast_classify_query(self, query: str) -> QueryClassification:
        """Lightning-fast query classification using pattern matching."""
        query_lower = query.lower()
        
        # Pattern-based classification
        scores = {}
        for intent, patterns in self.quick_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                scores[intent] = score / len(patterns)
        
        if scores:
            primary_intent = max(scores, key=scores.get)
            confidence = min(scores[primary_intent] * 2, 1.0)  # Boost confidence
        else:
            primary_intent = "general"
            confidence = 0.5
        
        return QueryClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=list(scores.keys())[:2],
            complexity_score=len(query.split()) / 20.0  # Simple complexity estimate
        )
    
    async def _comprehensive_classify_query(self, query: str) -> QueryClassification:
        """Full LLM-based classification for higher quality."""
        return await self.base_processor.classifier.classify_query(query)
    
    async def _create_routing_decision(self, 
                                     query: str, 
                                     classification: QueryClassification,
                                     context: Dict[str, Any]) -> RoutingDecision:
        """Create routing decision based on classification and mode."""
        
        if self.processing_mode == ProcessingMode.FAST:
            return await self._fast_route_query(query, classification)
        else:
            # Use the full routing logic from base processor
            routing_plan = await self.base_processor.process_query(query, context)
            return routing_plan["routing_decision"]
    
    async def _fast_route_query(self, query: str, classification: QueryClassification) -> RoutingDecision:
        """Fast routing based on simple rules."""
        from .query_router import DataSource, SearchStrategy
        
        # Simple routing rules
        if classification.primary_intent in ["technical", "troubleshooting"]:
            strategy = SearchStrategy.TECHNICAL_DEEP_DIVE
            sources = [DataSource.WIKI, DataSource.SLACK]
        elif classification.primary_intent == "comparison":
            strategy = SearchStrategy.COMPARATIVE_ANALYSIS  
            sources = [DataSource.WIKI]
        elif classification.primary_intent == "historical":
            strategy = SearchStrategy.CONVERSATIONAL_SEARCH
            sources = [DataSource.SLACK]
        else:
            strategy = SearchStrategy.HYBRID_SEARCH
            sources = [DataSource.WIKI, DataSource.SLACK, DataSource.WEB]
        
        return RoutingDecision(
            strategy=strategy,
            confidence=classification.confidence,
            data_sources=sources,
            search_params={
                "intent": classification.primary_intent,
                "processing_mode": "fast"
            }
        )
    
    def _should_use_query_enhancement(self, 
                                    classification: QueryClassification,
                                    routing_decision: RoutingDecision) -> bool:
        """Determine if query enhancement would be beneficial."""
        
        if self.processing_mode == ProcessingMode.FAST:
            return False  # Skip enhancement in fast mode
        
        # Use enhancement for complex or low-confidence queries
        return (
            classification.complexity_score > 0.5 or 
            classification.confidence < 0.7 or
            routing_decision.strategy.value in ["technical_deep_dive", "comparative_analysis"]
        )
    
    async def _fast_enhance_query(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Fast query enhancement without full HyDE processing."""
        
        # Simple keyword expansion based on intent
        expansion_terms = []
        
        if classification.primary_intent == "technical":
            expansion_terms = ["setup", "configuration", "implementation", "guide"]
        elif classification.primary_intent == "troubleshooting":
            expansion_terms = ["solution", "fix", "resolve", "error", "issue"]
        elif classification.primary_intent == "comparison":
            expansion_terms = ["difference", "comparison", "pros", "cons", "better"]
        
        enhanced_query = f"{query} {' '.join(expansion_terms[:2])}"
        
        return {
            "enhanced_query": enhanced_query,
            "expansion_terms": expansion_terms,
            "enhancement_type": "keyword_expansion"
        }

class StreamingQueryEngine:
    """
    Query engine wrapper that provides streaming capabilities for the entire RAG pipeline.
    """
    
    def __init__(self, 
                 base_query_engine,
                 fast_processor: FastQueryProcessor,
                 enable_streaming: bool = True):
        self.base_query_engine = base_query_engine
        self.fast_processor = fast_processor
        self.enable_streaming = enable_streaming
    
    async def query_streaming(self, query: str) -> AsyncGenerator[StreamingStep, None]:
        """Execute full RAG pipeline with streaming intermediate steps."""
        
        # Step 1: Agentic Processing
        async for step in self.fast_processor.process_query_streaming(query):
            yield step
            
            # Extract routing decision when processing completes
            if step.step_type == "completion":
                routing_decision = step.data["final_routing"]
        
        # Step 2: Retrieval
        yield StreamingStep("retrieval", "Document Retrieval", "starting", "Searching knowledge base...")
        
        retrieval_start = time.time()
        try:
            response = self.base_query_engine.query(query)
            retrieval_time = time.time() - retrieval_start
            
            yield StreamingStep(
                "retrieval", "Search Complete", "completed",
                f"Found {len(response.source_nodes)} relevant sources",
                {
                    "source_count": len(response.source_nodes),
                    "response_length": len(str(response))
                },
                duration=retrieval_time
            )
            
            # Step 3: Response Generation
            yield StreamingStep("generation", "Generating Response", "starting", "Synthesizing final answer...")
            
            generation_start = time.time()
            # Response is already generated, just report it
            generation_time = time.time() - generation_start
            
            yield StreamingStep(
                "generation", "Response Ready", "completed",
                "Response generated successfully",
                {"response": str(response), "sources": [str(node.node) for node in response.source_nodes]},
                duration=generation_time
            )
            
        except Exception as e:
            yield StreamingStep(
                "retrieval", "Retrieval Error", "error",
                f"Error during retrieval: {str(e)}",
                {"error_type": type(e).__name__}
            )
    
    def query(self, query: str):
        """Standard synchronous query interface."""
        return self.base_query_engine.query(query)