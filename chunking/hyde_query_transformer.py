"""
HyDE (Hypothetical Document Embeddings) Query Transformer

Implements query transformation by generating hypothetical documents
that would answer the query, then using those for improved retrieval.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

@dataclass
class HyDEResult:
    """Result from HyDE query transformation."""
    original_query: str
    hypothetical_documents: List[str]
    enhanced_queries: List[str]
    confidence_scores: List[float]

class HyDEStrategy(ABC):
    """Abstract base class for HyDE strategies."""
    
    @abstractmethod
    async def generate_hypothetical_documents(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate hypothetical documents for the query."""
        pass

class TechnicalHyDEStrategy(HyDEStrategy):
    """HyDE strategy optimized for technical queries."""
    
    def __init__(self, llm: OpenAI):
        self.llm = llm
    
    async def generate_hypothetical_documents(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate technical hypothetical documents."""
        domain = context.get('domain', 'general')
        
        prompts = [
            f"""You are a technical expert in {domain}. 
            Write a detailed technical document that would perfectly answer this question: "{query}"
            
            Include:
            1. Technical details and explanations
            2. Code examples if relevant
            3. Best practices and recommendations
            4. Common pitfalls to avoid
            
            Document:""",
            
            f"""You are writing documentation for {domain} developers.
            Create a comprehensive guide that addresses: "{query}"
            
            Structure your response as:
            - Overview and key concepts  
            - Step-by-step implementation
            - Configuration examples
            - Troubleshooting tips
            
            Guide:""",
            
            f"""You are answering a technical question in a {domain} forum.
            Provide a thorough response to: "{query}"
            
            Include:
            - Direct answer to the question
            - Supporting code or configuration
            - Links to relevant documentation
            - Alternative approaches
            
            Response:"""
        ]
        
        documents = []
        for prompt in prompts:
            try:
                response = await self.llm.acomplete(prompt)
                documents.append(response.text.strip())
            except Exception as e:
                logger.warning(f"Failed to generate hypothetical document: {e}")
                continue
        
        return documents

class ConversationalHyDEStrategy(HyDEStrategy):
    """HyDE strategy optimized for conversational/Slack queries."""
    
    def __init__(self, llm: OpenAI):
        self.llm = llm
    
    async def generate_hypothetical_documents(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate conversational hypothetical documents."""
        workspace = context.get('workspace', 'team')
        
        prompts = [
            f"""You are a helpful team member in a {workspace} workspace.
            Someone asked: "{query}"
            
            Write a helpful response that includes:
            - Direct answer to their question
            - Personal experience or examples
            - Suggestions for next steps
            - Offers to help further
            
            Response:""",
            
            f"""You are documenting a conversation where someone solved a problem.
            The original question was: "{query}"
            
            Write the conversation including:
            - The question and context
            - The solution that worked
            - Follow-up discussion
            - Lessons learned
            
            Conversation:""",
            
            f"""You are writing a FAQ entry based on this question: "{query}"
            
            Create a comprehensive FAQ entry with:
            - Clear restatement of the question  
            - Detailed answer with examples
            - Related questions and answers
            - When to seek additional help
            
            FAQ Entry:"""
        ]
        
        documents = []
        for prompt in prompts:
            try:
                response = await self.llm.acomplete(prompt)
                documents.append(response.text.strip())
            except Exception as e:
                logger.warning(f"Failed to generate hypothetical document: {e}")
                continue
        
        return documents

class WikiHyDEStrategy(HyDEStrategy):
    """HyDE strategy optimized for wiki/documentation queries."""
    
    def __init__(self, llm: OpenAI):
        self.llm = llm
    
    async def generate_hypothetical_documents(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate wiki-style hypothetical documents."""
        topic_area = context.get('topic_area', 'general')
        
        prompts = [
            f"""You are writing a comprehensive wiki article about {topic_area}.
            Create an article that thoroughly addresses: "{query}"
            
            Structure as a wiki article with:
            - Introduction and overview
            - Detailed sections with subheadings  
            - Examples and use cases
            - See also and references
            
            Article:""",
            
            f"""You are creating a tutorial for {topic_area}.
            Write a step-by-step tutorial that answers: "{query}"
            
            Include:
            - Prerequisites and setup
            - Detailed steps with explanations
            - Screenshots or code examples
            - Troubleshooting section
            
            Tutorial:""",
            
            f"""You are writing reference documentation for {topic_area}.
            Create comprehensive documentation for: "{query}"
            
            Include:
            - API/interface reference
            - Parameters and return values
            - Usage examples
            - Best practices
            
            Documentation:"""
        ]
        
        documents = []
        for prompt in prompts:
            try:
                response = await self.llm.acomplete(prompt)
                documents.append(response.text.strip())
            except Exception as e:
                logger.warning(f"Failed to generate hypothetical document: {e}")
                continue
        
        return documents

class HyDEQueryTransformer:
    """
    Main HyDE query transformer that generates hypothetical documents
    and enhanced queries for improved retrieval.
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large",
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        """Initialize HyDE transformer."""
        self.llm = OpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
        self.embedding_model = OpenAIEmbedding(model=embedding_model)
        
        # Strategy mapping
        self.strategies = {
            'technical': TechnicalHyDEStrategy(self.llm),
            'conversational': ConversationalHyDEStrategy(self.llm), 
            'wiki': WikiHyDEStrategy(self.llm)
        }
    
    async def transform_query(self, 
                            query: str, 
                            content_type: str = 'technical',
                            context: Optional[Dict[str, Any]] = None) -> HyDEResult:
        """
        Transform query using HyDE approach.
        
        Args:
            query: Original user query
            content_type: Type of content ('technical', 'conversational', 'wiki')
            context: Additional context for document generation
            
        Returns:
            HyDEResult with hypothetical documents and enhanced queries
        """
        if context is None:
            context = {}
        
        # Select appropriate strategy
        strategy = self.strategies.get(content_type, self.strategies['technical'])
        
        # Generate hypothetical documents
        hypothetical_docs = await strategy.generate_hypothetical_documents(query, context)
        
        # Generate enhanced queries from hypothetical documents
        enhanced_queries = await self._generate_enhanced_queries(query, hypothetical_docs)
        
        # Calculate confidence scores
        confidence_scores = await self._calculate_confidence_scores(query, hypothetical_docs)
        
        return HyDEResult(
            original_query=query,
            hypothetical_documents=hypothetical_docs,
            enhanced_queries=enhanced_queries,
            confidence_scores=confidence_scores
        )
    
    async def _generate_enhanced_queries(self, original_query: str, hypothetical_docs: List[str]) -> List[str]:
        """Generate enhanced queries from hypothetical documents."""
        enhanced_queries = []
        
        for doc in hypothetical_docs:
            if doc.strip():
                prompt = f"""Based on this document content, generate 2-3 alternative search queries 
                that would help find similar information to answer the original question: "{original_query}"
                
                Document excerpt: {doc[:500]}...
                
                Alternative queries (one per line):"""
                
                try:
                    response = await self.llm.acomplete(prompt)
                    queries = [q.strip() for q in response.text.split('\n') if q.strip()]
                    enhanced_queries.extend(queries)
                except Exception as e:
                    logger.warning(f"Failed to generate enhanced queries: {e}")
                    continue
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(enhanced_queries))
        return unique_queries[:5]  # Limit to top 5
    
    async def _calculate_confidence_scores(self, query: str, hypothetical_docs: List[str]) -> List[float]:
        """Calculate confidence scores for hypothetical documents."""
        scores = []
        
        for doc in hypothetical_docs:
            try:
                # Simple heuristic-based scoring
                score = self._calculate_document_quality_score(doc, query)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to calculate confidence score: {e}")
                scores.append(0.5)  # Default score
        
        return scores
    
    def _calculate_document_quality_score(self, doc: str, query: str) -> float:
        """Calculate quality score for a hypothetical document."""
        if not doc.strip():
            return 0.0
        
        # Length score (prefer medium-length documents)
        length_score = min(len(doc.split()) / 200, 1.0)
        
        # Query relevance score (keyword overlap)
        query_words = set(query.lower().split())
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        relevance_score = overlap / len(query_words) if query_words else 0.0
        
        # Structure score (prefer well-structured documents)
        structure_indicators = [
            len([line for line in doc.split('\n') if line.strip()]),  # Non-empty lines
            doc.count(':'),  # Colons indicate structure
            doc.count('-'),  # Bullet points
            doc.count('```'),  # Code blocks
        ]
        structure_score = min(sum(structure_indicators) / 10, 1.0)
        
        # Combine scores
        final_score = (length_score * 0.3 + relevance_score * 0.5 + structure_score * 0.2)
        return min(final_score, 1.0)
    
    def embed_hypothetical_documents(self, hypothetical_docs: List[str]) -> List[List[float]]:
        """Generate embeddings for hypothetical documents."""
        embeddings = []
        for doc in hypothetical_docs:
            if doc.strip():
                try:
                    embedding = self.embedding_model.get_text_embedding(doc)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed hypothetical document: {e}")
                    continue
        return embeddings
    
    async def transform_query_batch(self, 
                                  queries: List[str],
                                  content_type: str = 'technical',
                                  context: Optional[Dict[str, Any]] = None) -> List[HyDEResult]:
        """Transform multiple queries in batch."""
        tasks = [
            self.transform_query(query, content_type, context)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

class HyDEQueryEnhancer:
    """
    Enhanced query processor that combines original query with HyDE results
    for improved retrieval performance.
    """
    
    def __init__(self, hyde_transformer: HyDEQueryTransformer):
        self.hyde_transformer = hyde_transformer
    
    async def enhance_query_for_retrieval(self, 
                                        query: str,
                                        content_types: List[str] = ['technical'],
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance query for retrieval using HyDE approach.
        
        Returns:
            Dictionary with enhanced query components for retrieval
        """
        enhanced_components = {
            'original_query': query,
            'hypothetical_documents': [],
            'enhanced_queries': [],
            'search_embeddings': [],
            'confidence_scores': []
        }
        
        # Transform query for each content type
        for content_type in content_types:
            hyde_result = await self.hyde_transformer.transform_query(
                query, content_type, context
            )
            
            enhanced_components['hypothetical_documents'].extend(hyde_result.hypothetical_documents)
            enhanced_components['enhanced_queries'].extend(hyde_result.enhanced_queries)
            enhanced_components['confidence_scores'].extend(hyde_result.confidence_scores)
        
        # Generate embeddings for hypothetical documents
        if enhanced_components['hypothetical_documents']:
            embeddings = self.hyde_transformer.embed_hypothetical_documents(
                enhanced_components['hypothetical_documents']
            )
            enhanced_components['search_embeddings'] = embeddings
        
        # Generate combined search query
        all_queries = [query] + enhanced_components['enhanced_queries']
        enhanced_components['combined_query'] = self._combine_queries(all_queries)
        
        return enhanced_components
    
    def _combine_queries(self, queries: List[str]) -> str:
        """Combine multiple queries into a single enhanced query."""
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())
        
        # Combine with appropriate separators
        if len(unique_queries) == 1:
            return unique_queries[0]
        elif len(unique_queries) <= 3:
            return ' OR '.join(f'({q})' for q in unique_queries)
        else:
            # For many queries, use the original plus top 2 enhanced
            return f'({unique_queries[0]}) OR ({unique_queries[1]}) OR ({unique_queries[2]})'

# Example usage and integration helpers
class HyDEIntegration:
    """Helper class for integrating HyDE with existing RAG pipeline."""
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large"):
        self.query_enhancer = HyDEQueryEnhancer(
            HyDEQueryTransformer(llm_model, embedding_model)
        )
    
    async def enhance_search_query(self, 
                                 query: str,
                                 source_types: List[str] = ['technical', 'wiki'],
                                 domain_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Main integration point for enhancing queries in the RAG pipeline.
        
        Args:
            query: User's original query
            source_types: Types of content to optimize for
            domain_context: Domain-specific context (e.g., 'janelia', 'neuroscience')
            
        Returns:
            Enhanced query components for retrieval
        """
        context = {}
        if domain_context:
            context['domain'] = domain_context
            context['workspace'] = domain_context
            context['topic_area'] = domain_context
        
        return await self.query_enhancer.enhance_query_for_retrieval(
            query, source_types, context
        )
    
    def get_search_terms_for_hybrid_search(self, enhanced_query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract search terms optimized for hybrid (BM25 + vector) search.
        
        Returns:
            Dictionary with terms for different search modes
        """
        return {
            'bm25_query': enhanced_query_result['combined_query'],
            'vector_embeddings': enhanced_query_result['search_embeddings'],
            'fallback_queries': enhanced_query_result['enhanced_queries'][:3],
            'confidence_weights': enhanced_query_result['confidence_scores']
        }