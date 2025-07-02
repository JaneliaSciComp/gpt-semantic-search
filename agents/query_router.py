"""
Agentic Query Router and Classification System

Implements intelligent query routing, classification, and multi-step reasoning
for enhanced RAG retrieval. Routes queries to appropriate data sources and
processing strategies based on intent analysis.
"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of query intents for routing decisions."""
    TECHNICAL_QUESTION = "technical_question"        # How-to, implementation questions
    TROUBLESHOOTING = "troubleshooting"             # Error fixing, debugging
    INFORMATION_LOOKUP = "information_lookup"        # Finding specific information
    CONVERSATIONAL = "conversational"               # General discussion, opinions
    CODE_SEARCH = "code_search"                     # Looking for code examples
    COMPARISON = "comparison"                       # Comparing options/approaches
    PROCEDURAL = "procedural"                       # Step-by-step processes
    REFERENCE = "reference"                         # API docs, specifications
    HISTORICAL = "historical"                       # Past conversations, decisions

class DataSource(Enum):
    """Available data sources for query routing."""
    SLACK = "slack"
    WIKI = "wiki" 
    WEB = "web"
    ALL = "all"

class QueryComplexity(Enum):
    """Query complexity levels for processing strategy."""
    SIMPLE = "simple"          # Single-step lookup
    MODERATE = "moderate"      # Multi-step reasoning
    COMPLEX = "complex"        # Requires decomposition and tools

@dataclass
class QueryClassification:
    """Result of query classification."""
    intent: QueryIntent
    complexity: QueryComplexity
    data_sources: List[DataSource]
    confidence: float
    reasoning: str
    keywords: List[str]
    entities: List[str]
    temporal_context: Optional[str] = None

@dataclass
class RoutingDecision:
    """Routing decision with strategy and parameters."""
    strategy: str
    data_sources: List[DataSource]
    search_params: Dict[str, Any]
    processing_steps: List[str]
    confidence: float
    fallback_strategies: List[str]

class QueryClassifier:
    """
    Intelligent query classifier that analyzes user queries to determine
    intent, complexity, and optimal routing strategy.
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        
        # Pattern matching for quick classification
        self.intent_patterns = {
            QueryIntent.TECHNICAL_QUESTION: [
                r'\bhow\s+(to|do|can)\b', r'\bwhat\s+is\b', r'\bexplain\b',
                r'\bimplement\b', r'\bconfigure\b', r'\bsetup\b'
            ],
            QueryIntent.TROUBLESHOOTING: [
                r'\berror\b', r'\bfail\b', r'\bissue\b', r'\bproblem\b',
                r'\bbug\b', r'\bbroken\b', r'\bnot\s+work\b', r'\bfix\b'
            ],
            QueryIntent.CODE_SEARCH: [
                r'\bcode\b', r'\bexample\b', r'\bsnippet\b', r'\bfunction\b',
                r'\bclass\b', r'\bmethod\b', r'\bapi\b', r'\blibrary\b'
            ],
            QueryIntent.CONVERSATIONAL: [
                r'\bwhat\s+do\s+you\s+think\b', r'\bopinion\b', r'\bprefer\b',
                r'\bbest\s+practice\b', r'\brecommend\b', r'\badvice\b'
            ],
            QueryIntent.COMPARISON: [
                r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bdifference\b',
                r'\bbetter\b', r'\bwhich\s+one\b', r'\balternative\b'
            ],
            QueryIntent.HISTORICAL: [
                r'\bwhen\s+did\b', r'\bprevious\b', r'\blast\s+time\b',
                r'\bearlier\b', r'\bago\b', r'\bhistory\b'
            ]
        }
        
        self.source_patterns = {
            DataSource.SLACK: [
                r'\bslack\b', r'\bchat\b', r'\bmessage\b', r'\bconversation\b',
                r'\bdiscussion\b', r'\bsaid\b', r'\bmentioned\b'
            ],
            DataSource.WIKI: [
                r'\bdocumentation\b', r'\bdocs\b', r'\bwiki\b', r'\bguide\b',
                r'\btutorial\b', r'\bmanual\b', r'\bspecification\b'
            ],
            DataSource.WEB: [
                r'\bwebsite\b', r'\bpage\b', r'\bonline\b', r'\burl\b',
                r'\blink\b', r'\bjanelia\.org\b'
            ]
        }
    
    async def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryClassification:
        """Classify a query to determine intent and routing strategy."""
        # Quick pattern-based classification
        pattern_classification = self._classify_with_patterns(query)
        
        # LLM-based detailed classification
        llm_classification = await self._classify_with_llm(query, context)
        
        # Combine results
        final_classification = self._combine_classifications(
            query, pattern_classification, llm_classification
        )
        
        return final_classification
    
    def _classify_with_patterns(self, query: str) -> Dict[str, Any]:
        """Quick pattern-based classification."""
        query_lower = query.lower()
        
        # Intent classification
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                intent_scores[intent] = score
        
        # Data source hints
        source_scores = {}
        for source, patterns in self.source_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                source_scores[source] = score
        
        # Complexity heuristics
        complexity = QueryComplexity.SIMPLE
        if len(query.split()) > 15 or '?' in query.count('?') > 1:
            complexity = QueryComplexity.MODERATE
        if any(word in query_lower for word in ['multiple', 'several', 'compare', 'analyze']):
            complexity = QueryComplexity.COMPLEX
        
        return {
            'intent_scores': intent_scores,
            'source_scores': source_scores,
            'complexity': complexity,
            'confidence': 0.7 if intent_scores else 0.3
        }
    
    async def _classify_with_llm(self, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """LLM-based detailed classification."""
        context_str = ""
        if context:
            context_str = f"Context: {context}\n\n"
        
        prompt = f"""
        {context_str}Analyze this query and provide detailed classification:
        
        Query: "{query}"
        
        Provide analysis in the following format:
        
        INTENT: [technical_question|troubleshooting|information_lookup|conversational|code_search|comparison|procedural|reference|historical]
        
        COMPLEXITY: [simple|moderate|complex]
        
        DATA_SOURCES: [slack|wiki|web|all] (comma-separated, order by relevance)
        
        KEYWORDS: [key terms for search] (comma-separated)
        
        ENTITIES: [named entities, tools, technologies] (comma-separated)
        
        TEMPORAL: [if query has time context, specify: recent|historical|specific_date|none]
        
        REASONING: [brief explanation of classification decisions]
        
        CONFIDENCE: [0.0-1.0]
        """
        
        try:
            response = await self.llm.acomplete(prompt)
            return self._parse_llm_response(response.text)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return {'confidence': 0.3}
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        result = {}
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'intent':
                        try:
                            result['intent'] = QueryIntent(value)
                        except ValueError:
                            result['intent'] = QueryIntent.INFORMATION_LOOKUP
                    elif key == 'complexity':
                        try:
                            result['complexity'] = QueryComplexity(value)
                        except ValueError:
                            result['complexity'] = QueryComplexity.SIMPLE
                    elif key == 'data_sources':
                        sources = []
                        for source in value.split(','):
                            try:
                                sources.append(DataSource(source.strip()))
                            except ValueError:
                                continue
                        result['data_sources'] = sources or [DataSource.ALL]
                    elif key == 'keywords':
                        result['keywords'] = [kw.strip() for kw in value.split(',') if kw.strip()]
                    elif key == 'entities':
                        result['entities'] = [ent.strip() for ent in value.split(',') if ent.strip()]
                    elif key == 'temporal':
                        result['temporal'] = value if value != 'none' else None
                    elif key == 'reasoning':
                        result['reasoning'] = value
                    elif key == 'confidence':
                        try:
                            result['confidence'] = float(value)
                        except ValueError:
                            result['confidence'] = 0.5
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            result['confidence'] = 0.3
        
        return result
    
    def _combine_classifications(self, 
                               query: str,
                               pattern_result: Dict,
                               llm_result: Dict) -> QueryClassification:
        """Combine pattern and LLM classifications."""
        
        # Determine intent (prefer LLM if high confidence)
        if llm_result.get('confidence', 0) > 0.7 and 'intent' in llm_result:
            intent = llm_result['intent']
        elif pattern_result['intent_scores']:
            intent = max(pattern_result['intent_scores'].items(), key=lambda x: x[1])[0]
        else:
            intent = llm_result.get('intent', QueryIntent.INFORMATION_LOOKUP)
        
        # Determine complexity
        complexity = llm_result.get('complexity', pattern_result['complexity'])
        
        # Determine data sources
        if 'data_sources' in llm_result:
            data_sources = llm_result['data_sources']
        elif pattern_result['source_scores']:
            data_sources = [max(pattern_result['source_scores'].items(), key=lambda x: x[1])[0]]
        else:
            data_sources = [DataSource.ALL]
        
        # Combine confidence scores
        pattern_conf = pattern_result.get('confidence', 0.3)
        llm_conf = llm_result.get('confidence', 0.3)
        combined_confidence = (pattern_conf + llm_conf) / 2
        
        return QueryClassification(
            intent=intent,
            complexity=complexity,
            data_sources=data_sources,
            confidence=combined_confidence,
            reasoning=llm_result.get('reasoning', 'Pattern-based classification'),
            keywords=llm_result.get('keywords', self._extract_keywords(query)),
            entities=llm_result.get('entities', []),
            temporal_context=llm_result.get('temporal')
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query using simple heuristics."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10

class QueryRouter:
    """
    Main query router that makes routing decisions based on classification
    and available data sources.
    """
    
    def __init__(self, 
                 classifier: QueryClassifier,
                 available_sources: List[DataSource] = None):
        self.classifier = classifier
        self.available_sources = available_sources or [DataSource.SLACK, DataSource.WIKI, DataSource.WEB]
        
        # Strategy definitions
        self.strategies = {
            'direct_search': self._direct_search_strategy,
            'multi_source_search': self._multi_source_search_strategy,
            'conversational_search': self._conversational_search_strategy,
            'technical_deep_dive': self._technical_deep_dive_strategy,
            'troubleshooting_workflow': self._troubleshooting_workflow_strategy,
            'comparative_analysis': self._comparative_analysis_strategy,
            'historical_lookup': self._historical_lookup_strategy
        }
    
    async def route_query(self, 
                         query: str, 
                         context: Optional[Dict] = None) -> RoutingDecision:
        """Route a query to the appropriate strategy and data sources."""
        
        # Classify the query
        classification = await self.classifier.classify_query(query, context)
        
        # Determine routing strategy
        strategy = self._select_strategy(classification)
        
        # Configure search parameters
        search_params = self._configure_search_params(classification)
        
        # Define processing steps
        processing_steps = self._define_processing_steps(classification, strategy)
        
        # Select fallback strategies
        fallback_strategies = self._select_fallback_strategies(classification, strategy)
        
        # Filter available data sources
        filtered_sources = [
            source for source in classification.data_sources 
            if source in self.available_sources or source == DataSource.ALL
        ]
        if not filtered_sources:
            filtered_sources = self.available_sources
        
        return RoutingDecision(
            strategy=strategy,
            data_sources=filtered_sources,
            search_params=search_params,
            processing_steps=processing_steps,
            confidence=classification.confidence,
            fallback_strategies=fallback_strategies
        )
    
    def _select_strategy(self, classification: QueryClassification) -> str:
        """Select the best strategy based on classification."""
        intent = classification.intent
        complexity = classification.complexity
        
        # Intent-based strategy selection
        if intent == QueryIntent.TECHNICAL_QUESTION:
            return 'technical_deep_dive' if complexity == QueryComplexity.COMPLEX else 'direct_search'
        elif intent == QueryIntent.TROUBLESHOOTING:
            return 'troubleshooting_workflow'
        elif intent == QueryIntent.CONVERSATIONAL:
            return 'conversational_search'
        elif intent == QueryIntent.COMPARISON:
            return 'comparative_analysis'
        elif intent == QueryIntent.HISTORICAL:
            return 'historical_lookup'
        elif intent in [QueryIntent.CODE_SEARCH, QueryIntent.REFERENCE]:
            return 'direct_search'
        else:
            return 'multi_source_search' if complexity != QueryComplexity.SIMPLE else 'direct_search'
    
    def _configure_search_params(self, classification: QueryClassification) -> Dict[str, Any]:
        """Configure search parameters based on classification."""
        params = {
            'keywords': classification.keywords,
            'entities': classification.entities,
            'intent': classification.intent.value,
            'complexity': classification.complexity.value,
            'temporal_context': classification.temporal_context
        }
        
        # Intent-specific parameters
        if classification.intent == QueryIntent.CODE_SEARCH:
            params['prefer_code_blocks'] = True
            params['code_languages'] = self._extract_code_languages(classification.keywords)
        
        if classification.intent == QueryIntent.TROUBLESHOOTING:
            params['error_keywords'] = [kw for kw in classification.keywords if any(err in kw.lower() for err in ['error', 'fail', 'issue', 'problem'])]
        
        if classification.intent == QueryIntent.HISTORICAL:
            params['time_weight'] = 0.8  # Prioritize recent content
            params['sort_by_time'] = True
        
        # Complexity-based parameters
        if classification.complexity == QueryComplexity.COMPLEX:
            params['multi_hop'] = True
            params['decompose_query'] = True
            params['max_results_per_source'] = 20
        else:
            params['max_results_per_source'] = 10
        
        return params
    
    def _extract_code_languages(self, keywords: List[str]) -> List[str]:
        """Extract programming languages from keywords."""
        languages = ['python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust', 'scala', 'r', 'matlab', 'sql', 'bash', 'shell']
        detected = []
        for keyword in keywords:
            for lang in languages:
                if lang in keyword.lower():
                    detected.append(lang)
        return detected
    
    def _define_processing_steps(self, 
                               classification: QueryClassification,
                               strategy: str) -> List[str]:
        """Define processing steps for the selected strategy."""
        base_steps = ['retrieve_documents', 'rank_results', 'generate_response']
        
        # Strategy-specific steps
        if strategy == 'technical_deep_dive':
            return ['analyze_query', 'retrieve_documents', 'extract_technical_details', 'cross_reference_sources', 'synthesize_comprehensive_response']
        
        elif strategy == 'troubleshooting_workflow':
            return ['identify_error_context', 'search_similar_issues', 'retrieve_solutions', 'rank_by_relevance', 'format_troubleshooting_guide']
        
        elif strategy == 'conversational_search':
            return ['retrieve_conversations', 'extract_discussion_context', 'identify_consensus', 'generate_conversational_response']
        
        elif strategy == 'comparative_analysis':
            return ['decompose_comparison', 'retrieve_multiple_topics', 'extract_comparison_points', 'synthesize_comparison_table', 'generate_recommendation']
        
        elif strategy == 'historical_lookup':
            return ['time_filtered_search', 'sort_by_recency', 'extract_timeline', 'generate_historical_summary']
        
        elif strategy == 'multi_source_search':
            return ['parallel_source_search', 'deduplicate_results', 'cross_validate_information', 'merge_perspectives', 'generate_unified_response']
        
        else:  # direct_search
            return base_steps
    
    def _select_fallback_strategies(self, 
                                  classification: QueryClassification,
                                  primary_strategy: str) -> List[str]:
        """Select fallback strategies if primary fails."""
        fallbacks = []
        
        if primary_strategy != 'direct_search':
            fallbacks.append('direct_search')
        
        if primary_strategy != 'multi_source_search' and classification.complexity != QueryComplexity.SIMPLE:
            fallbacks.append('multi_source_search')
        
        if classification.intent == QueryIntent.TECHNICAL_QUESTION and primary_strategy != 'conversational_search':
            fallbacks.append('conversational_search')
        
        return fallbacks[:2]  # Limit to 2 fallback strategies
    
    # Strategy implementation methods (to be integrated with retrieval system)
    def _direct_search_strategy(self, **kwargs):
        """Simple direct search strategy."""
        pass
    
    def _multi_source_search_strategy(self, **kwargs):
        """Multi-source parallel search strategy.""" 
        pass
    
    def _conversational_search_strategy(self, **kwargs):
        """Conversational context-aware search strategy."""
        pass
    
    def _technical_deep_dive_strategy(self, **kwargs):
        """Deep technical analysis strategy."""
        pass
    
    def _troubleshooting_workflow_strategy(self, **kwargs):
        """Systematic troubleshooting workflow."""
        pass
    
    def _comparative_analysis_strategy(self, **kwargs):
        """Comparative analysis strategy."""
        pass
    
    def _historical_lookup_strategy(self, **kwargs):
        """Historical/temporal search strategy."""
        pass

class AgenticQueryProcessor:
    """
    Complete agentic query processing system that orchestrates
    classification, routing, and execution strategies.
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 available_sources: List[DataSource] = None):
        self.classifier = QueryClassifier(llm_model)
        self.router = QueryRouter(self.classifier, available_sources)
        self.llm = OpenAI(model=llm_model, temperature=0.3)
    
    async def process_query(self, 
                          query: str,
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query through the complete agentic pipeline.
        
        Returns:
            Dictionary with routing decision and processing plan
        """
        # Route the query
        routing_decision = await self.router.route_query(query, context)
        
        # Create processing plan
        processing_plan = {
            'original_query': query,
            'routing_decision': routing_decision,
            'execution_plan': self._create_execution_plan(routing_decision),
            'monitoring_points': self._define_monitoring_points(routing_decision)
        }
        
        return processing_plan
    
    def _create_execution_plan(self, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Create detailed execution plan from routing decision."""
        return {
            'strategy': routing_decision.strategy,
            'data_sources': [source.value for source in routing_decision.data_sources],
            'search_parameters': routing_decision.search_params,
            'processing_steps': routing_decision.processing_steps,
            'fallback_strategies': routing_decision.fallback_strategies,
            'confidence_threshold': routing_decision.confidence
        }
    
    def _define_monitoring_points(self, routing_decision: RoutingDecision) -> List[str]:
        """Define monitoring points for execution tracking."""
        return [
            'query_classification_complete',
            'data_retrieval_started',
            'results_ranking_complete',
            'response_generation_started', 
            'response_validation_complete'
        ]
    
    async def decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        prompt = f"""
        Decompose this complex query into 2-4 simpler, more focused sub-queries that together would provide a complete answer:
        
        Original query: "{query}"
        
        Sub-queries (one per line):
        """
        
        try:
            response = await self.llm.acomplete(prompt)
            sub_queries = [q.strip() for q in response.text.split('\n') if q.strip()]
            return sub_queries[:4]  # Limit to 4 sub-queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]  # Return original query as fallback