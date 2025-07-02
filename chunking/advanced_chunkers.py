"""
Advanced chunking strategies for modern RAG systems.

Implements hierarchical, syntax-based, and semantic chunking approaches
for different content types (code, markdown, Slack conversations, etc.).
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
from abc import ABC, abstractmethod

class ContentType(Enum):
    """Content type enumeration for different chunking strategies."""
    MARKDOWN = "markdown"
    CODE = "code" 
    SLACK = "slack"
    WEB = "web"
    PLAIN_TEXT = "plain_text"

@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    chunk_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    level: int = 0
    content_type: ContentType = ContentType.PLAIN_TEXT
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    semantic_score: float = 0.0

@dataclass 
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.metadata.chunk_id,
            "parent_id": self.metadata.parent_id,
            "children_ids": self.metadata.children_ids or [],
            "level": self.metadata.level,
            "content_type": self.metadata.content_type.value,
            "start_char": self.metadata.start_char,
            "end_char": self.metadata.end_char,
            "token_count": self.metadata.token_count,
            "semantic_score": self.metadata.semantic_score
        }

class BaseChunker(ABC):
    """Abstract base class for document chunkers."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str, **kwargs) -> List[DocumentChunk]:
        """Chunk document into smaller pieces."""
        pass

class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunker that creates parent-child relationships.
    
    Creates multiple levels:
    - Level 0: Full document 
    - Level 1: Major sections (headers, chapters)
    - Level 2: Subsections 
    - Level 3: Paragraphs/chunks
    """
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 min_chunk_size: int = 50,
                 overlap_size: int = 50,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
    
    def chunk(self, text: str, doc_id: str, content_type: ContentType = ContentType.PLAIN_TEXT) -> List[DocumentChunk]:
        """Create hierarchical chunks from document."""
        chunks = []
        
        # Level 0: Full document
        root_chunk = DocumentChunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=f"{doc_id}_root",
                level=0,
                content_type=content_type,
                start_char=0,
                end_char=len(text),
                token_count=self.count_tokens(text)
            )
        )
        chunks.append(root_chunk)
        
        # Level 1+: Hierarchical breakdown based on content type
        if content_type == ContentType.MARKDOWN:
            chunks.extend(self._chunk_markdown_hierarchical(text, doc_id, root_chunk.metadata.chunk_id))
        elif content_type == ContentType.CODE:
            chunks.extend(self._chunk_code_hierarchical(text, doc_id, root_chunk.metadata.chunk_id))
        elif content_type == ContentType.SLACK:
            chunks.extend(self._chunk_slack_hierarchical(text, doc_id, root_chunk.metadata.chunk_id))
        else:
            chunks.extend(self._chunk_text_hierarchical(text, doc_id, root_chunk.metadata.chunk_id))
        
        return chunks
    
    def _chunk_markdown_hierarchical(self, text: str, doc_id: str, parent_id: str) -> List[DocumentChunk]:
        """Hierarchical chunking for markdown documents."""
        chunks = []
        
        # Split by headers (# ## ### etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        current_section = []
        current_header_level = 0
        section_start = 0
        
        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Process previous section
                if current_section:
                    section_text = '\n'.join(current_section)
                    if len(section_text.strip()) > self.min_chunk_size:
                        chunks.extend(self._create_section_chunks(
                            section_text, doc_id, parent_id, 
                            current_header_level + 1, section_start
                        ))
                
                # Start new section
                header_level = len(header_match.group(1))
                current_header_level = header_level
                current_section = [line]
                section_start = sum(len(l) + 1 for l in lines[:i])
            else:
                current_section.append(line)
        
        # Process final section
        if current_section:
            section_text = '\n'.join(current_section)
            if len(section_text.strip()) > self.min_chunk_size:
                chunks.extend(self._create_section_chunks(
                    section_text, doc_id, parent_id,
                    current_header_level + 1, section_start
                ))
        
        return chunks
    
    def _chunk_code_hierarchical(self, text: str, doc_id: str, parent_id: str) -> List[DocumentChunk]:
        """Hierarchical chunking for code documents."""
        chunks = []
        
        # Split by classes and functions
        class_pattern = r'^(class\s+\w+.*?:)'
        function_pattern = r'^(\s*def\s+\w+.*?:)'
        
        lines = text.split('\n')
        current_block = []
        block_start = 0
        block_level = 1
        
        for i, line in enumerate(lines):
            if re.match(class_pattern, line) or re.match(function_pattern, line):
                # Process previous block
                if current_block:
                    block_text = '\n'.join(current_block)
                    if len(block_text.strip()) > self.min_chunk_size:
                        chunks.extend(self._create_section_chunks(
                            block_text, doc_id, parent_id,
                            block_level, block_start
                        ))
                
                # Start new block
                current_block = [line]
                block_start = sum(len(l) + 1 for l in lines[:i])
                block_level = 1 if re.match(class_pattern, line) else 2
            else:
                current_block.append(line)
        
        # Process final block
        if current_block:
            block_text = '\n'.join(current_block)
            if len(block_text.strip()) > self.min_chunk_size:
                chunks.extend(self._create_section_chunks(
                    block_text, doc_id, parent_id,
                    block_level, block_start
                ))
        
        return chunks
    
    def _chunk_slack_hierarchical(self, text: str, doc_id: str, parent_id: str) -> List[DocumentChunk]:
        """Hierarchical chunking for Slack conversations."""
        chunks = []
        
        # Split by conversation threads
        thread_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        messages = re.split(thread_pattern, text)[1:]  # Skip first empty element
        
        current_thread = []
        thread_start = 0
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                timestamp = messages[i]
                message = messages[i + 1]
                
                # Create message chunk
                message_text = f"{timestamp}{message}"
                if len(message_text.strip()) > self.min_chunk_size:
                    chunks.extend(self._create_section_chunks(
                        message_text, doc_id, parent_id,
                        1, thread_start
                    ))
                
                thread_start += len(message_text)
        
        return chunks
    
    def _chunk_text_hierarchical(self, text: str, doc_id: str, parent_id: str) -> List[DocumentChunk]:
        """Hierarchical chunking for plain text."""
        chunks = []
        
        # Split by paragraphs, then by sentences
        paragraphs = text.split('\n\n')
        char_offset = 0
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > self.min_chunk_size:
                chunks.extend(self._create_section_chunks(
                    paragraph, doc_id, parent_id,
                    1, char_offset
                ))
            char_offset += len(paragraph) + 2
        
        return chunks
    
    def _create_section_chunks(self, text: str, doc_id: str, parent_id: str, 
                              level: int, start_offset: int) -> List[DocumentChunk]:
        """Create chunks from a section of text."""
        chunks = []
        token_count = self.count_tokens(text)
        
        if token_count <= self.max_chunk_size:
            # Single chunk
            chunk_id = f"{doc_id}_L{level}_{len(chunks)}"
            chunk = DocumentChunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    level=level,
                    start_char=start_offset,
                    end_char=start_offset + len(text),
                    token_count=token_count
                )
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            sentences = self._split_into_sentences(text)
            current_chunk = []
            current_tokens = 0
            chunk_start = start_offset
            overlap_text = ""
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = f"{doc_id}_L{level}_{len(chunks)}"
                    chunk = DocumentChunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            chunk_id=chunk_id,
                            parent_id=parent_id,
                            level=level,
                            start_char=chunk_start,
                            end_char=chunk_start + len(chunk_text),
                            token_count=current_tokens
                        )
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    if self.overlap_size > 0 and len(current_chunk) > 1:
                        overlap_text = ' '.join(current_chunk[-1:])
                        current_chunk = [overlap_text, sentence]
                        current_tokens = self.count_tokens(overlap_text) + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    
                    chunk_start += len(chunk_text) - (len(overlap_text) if self.overlap_size > 0 else 0)
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_L{level}_{len(chunks)}"
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        parent_id=parent_id,  
                        level=level,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        token_count=current_tokens
                    )
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with more sophisticated methods
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

class SyntaxAwareChunker(BaseChunker):
    """
    Syntax-aware chunker that respects content structure.
    
    Uses different strategies based on content type:
    - Code: Function/class boundaries
    - Markdown: Header structure  
    - Slack: Message boundaries
    - Web: HTML structure
    """
    
    def __init__(self, 
                 target_chunk_size: int = 512,
                 max_chunk_size: int = 800,
                 min_chunk_size: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, doc_id: str, content_type: ContentType = ContentType.PLAIN_TEXT) -> List[DocumentChunk]:
        """Chunk text using syntax-aware strategies."""
        if content_type == ContentType.CODE:
            return self._chunk_code_syntax(text, doc_id)
        elif content_type == ContentType.MARKDOWN:
            return self._chunk_markdown_syntax(text, doc_id)
        elif content_type == ContentType.SLACK:
            return self._chunk_slack_syntax(text, doc_id)
        elif content_type == ContentType.WEB:
            return self._chunk_web_syntax(text, doc_id)
        else:
            return self._chunk_semantic_boundaries(text, doc_id)
    
    def _chunk_code_syntax(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk code respecting function/class boundaries."""
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_tokens = 0
        brace_level = 0
        in_function = False
        chunk_start = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # Track code structure
            brace_level += line.count('{') - line.count('}')
            
            if re.match(r'^\s*(def|class|function)\s+', line):
                in_function = True
            
            # Check if we should break the chunk
            should_break = (
                current_tokens + line_tokens > self.target_chunk_size and
                brace_level == 0 and
                not in_function and
                len(current_chunk) > 0
            )
            
            if should_break:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), chunk_start))
                
                current_chunk = [line]
                current_tokens = line_tokens
                chunk_start = sum(len(l) + 1 for l in lines[:i])
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
            
            if brace_level == 0:
                in_function = False
        
        # Final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), chunk_start))
        
        return chunks
    
    def _chunk_markdown_syntax(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk markdown respecting header structure."""
        chunks = []
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        char_offset = 0
        
        for i, section in enumerate(sections):
            if section.strip():
                section_tokens = self.count_tokens(section)
                
                if section_tokens <= self.max_chunk_size:
                    chunks.append(self._create_chunk(section, doc_id, i, char_offset))
                else:
                    # Split large sections
                    sub_chunks = self._split_large_section(section, doc_id, i, char_offset)
                    chunks.extend(sub_chunks)
                
                char_offset += len(section) + 1
        
        return chunks
    
    def _chunk_slack_syntax(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk Slack messages respecting conversation flow."""
        chunks = []
        
        # Split by message timestamps
        message_pattern = r'\n(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        messages = re.split(message_pattern, text)
        
        current_chunk_messages = []
        current_tokens = 0
        char_offset = 0
        
        for message in messages:
            if message.strip():
                message_tokens = self.count_tokens(message)
                
                if current_tokens + message_tokens > self.target_chunk_size and current_chunk_messages:
                    chunk_text = '\n'.join(current_chunk_messages)
                    chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                    
                    current_chunk_messages = [message]
                    current_tokens = message_tokens
                else:
                    current_chunk_messages.append(message)
                    current_tokens += message_tokens
                
                char_offset += len(message) + 1
        
        # Final chunk
        if current_chunk_messages:
            chunk_text = '\n'.join(current_chunk_messages)
            chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _chunk_web_syntax(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk web content respecting HTML structure."""
        # Simple implementation - can be enhanced with HTML parsing
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        char_offset = 0
        
        current_chunk_paragraphs = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            if paragraph.strip():
                para_tokens = self.count_tokens(paragraph)
                
                if current_tokens + para_tokens > self.target_chunk_size and current_chunk_paragraphs:
                    chunk_text = '\n\n'.join(current_chunk_paragraphs)
                    chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                    
                    current_chunk_paragraphs = [paragraph]
                    current_tokens = para_tokens
                else:
                    current_chunk_paragraphs.append(paragraph)
                    current_tokens += para_tokens
                
                char_offset += len(paragraph) + 2
        
        # Final chunk  
        if current_chunk_paragraphs:
            chunk_text = '\n\n'.join(current_chunk_paragraphs)
            chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _chunk_semantic_boundaries(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk based on semantic boundaries (sentences, paragraphs)."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        char_offset = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            char_offset += len(sentence) + 1
        
        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _split_large_section(self, text: str, doc_id: str, section_id: int, start_offset: int) -> List[DocumentChunk]:
        """Split large sections into smaller chunks."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        char_offset = start_offset
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_s{section_id}_{len(chunks)}"
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        start_char=char_offset - len(chunk_text),
                        end_char=char_offset,
                        token_count=current_tokens
                    )
                ))
                
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            char_offset += len(sentence) + 1
        
        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_s{section_id}_{len(chunks)}"
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    start_char=char_offset - len(chunk_text),
                    end_char=char_offset,
                    token_count=current_tokens
                )
            ))
        
        return chunks
    
    def _create_chunk(self, text: str, doc_id: str, chunk_num: int, start_char: int) -> DocumentChunk:
        """Create a document chunk with metadata."""
        chunk_id = f"{doc_id}_chunk_{chunk_num}"
        return DocumentChunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=start_char + len(text),
                token_count=self.count_tokens(text)
            )
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

class AdaptiveChunker(BaseChunker):
    """
    Adaptive chunker that dynamically adjusts chunk size based on content characteristics.
    
    Uses content analysis to determine optimal chunk sizes:
    - Dense technical content: Smaller chunks (256 tokens)
    - Narrative content: Larger chunks (768 tokens)  
    - Code: Function-based chunking
    - Conversations: Message-based chunking
    """
    
    def __init__(self, 
                 base_chunk_size: int = 512,
                 min_chunk_size: int = 128,
                 max_chunk_size: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, doc_id: str, content_type: ContentType = ContentType.PLAIN_TEXT) -> List[DocumentChunk]:
        """Adaptively chunk text based on content characteristics."""
        # Analyze content to determine optimal chunk size
        optimal_size = self._calculate_optimal_chunk_size(text, content_type)
        
        # Use appropriate chunking strategy
        if content_type == ContentType.CODE:
            return self._adaptive_code_chunking(text, doc_id, optimal_size)
        elif content_type == ContentType.SLACK:
            return self._adaptive_conversation_chunking(text, doc_id, optimal_size)
        else:
            return self._adaptive_semantic_chunking(text, doc_id, optimal_size)
    
    def _calculate_optimal_chunk_size(self, text: str, content_type: ContentType) -> int:
        """Calculate optimal chunk size based on content analysis."""
        base_size = self.base_chunk_size
        
        # Content type adjustments
        if content_type == ContentType.CODE:
            base_size = int(base_size * 0.8)  # Smaller for dense code
        elif content_type == ContentType.SLACK:
            base_size = int(base_size * 1.2)  # Larger for conversations
        
        # Content density analysis
        density_score = self._analyze_content_density(text)
        size_adjustment = 1.0 - (density_score * 0.3)  # Reduce size for dense content
        
        optimal_size = int(base_size * size_adjustment)
        return max(self.min_chunk_size, min(optimal_size, self.max_chunk_size))
    
    def _analyze_content_density(self, text: str) -> float:
        """
        Analyze content density to inform chunking decisions.
        Returns score from 0.0 (low density) to 1.0 (high density).
        """
        # Simple heuristics - can be enhanced with ML models
        words = text.split()
        if not words:
            return 0.0
        
        # Technical indicators
        technical_terms = len([w for w in words if any(c in w for c in ['_', '/', '\\', '()', '{}', '[]'])])
        technical_ratio = technical_terms / len(words)
        
        # Sentence length variance (dense content has more varied sentence lengths)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                variance_score = min(variance / 100, 1.0)
            else:
                variance_score = 0.0
        else:
            variance_score = 0.0
        
        # Combine scores
        density_score = (technical_ratio * 0.6 + variance_score * 0.4)
        return min(density_score, 1.0)
    
    def _adaptive_code_chunking(self, text: str, doc_id: str, target_size: int) -> List[DocumentChunk]:
        """Adaptive chunking for code content."""
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_tokens = 0
        char_offset = 0
        in_multiline_construct = False
        brace_level = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # Track code structure
            brace_level += line.count('{') - line.count('}')
            brace_level += line.count('[') - line.count(']')
            brace_level += line.count('(') - line.count(')')
            
            # Check for natural break points
            is_break_point = (
                brace_level == 0 and
                not in_multiline_construct and
                (line.strip() == '' or 
                 line.strip().startswith('#') or
                 re.match(r'^\s*(def|class|if|for|while)', line))
            )
            
            should_break = (
                current_tokens + line_tokens > target_size and
                len(current_chunk) > 0 and
                is_break_point
            )
            
            if should_break:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
            
            char_offset += len(line) + 1
        
        # Final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _adaptive_conversation_chunking(self, text: str, doc_id: str, target_size: int) -> List[DocumentChunk]:
        """Adaptive chunking for conversation content."""
        chunks = []
        
        # Split by conversation markers
        message_pattern = r'\n(?=\d{4}-\d{2}-\d{2}|\[|\w+:)'
        segments = re.split(message_pattern, text)
        
        current_chunk = []
        current_tokens = 0
        char_offset = 0
        
        for segment in segments:
            if segment.strip():
                segment_tokens = self.count_tokens(segment)
                
                # Check if this completes a conversational unit
                is_conversation_end = (
                    segment.strip().endswith('?') or
                    segment.strip().endswith('.') or
                    len(re.findall(r'\n', segment)) > 2
                )
                
                should_break = (
                    current_tokens + segment_tokens > target_size and
                    len(current_chunk) > 0 and
                    is_conversation_end
                )
                
                if should_break:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                    
                    current_chunk = [segment]
                    current_tokens = segment_tokens
                else:
                    current_chunk.append(segment)
                    current_tokens += segment_tokens
                
                char_offset += len(segment) + 1
        
        # Final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _adaptive_semantic_chunking(self, text: str, doc_id: str, target_size: int) -> List[DocumentChunk]:
        """Adaptive semantic chunking for general content."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        char_offset = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Calculate semantic coherence (simple heuristic)
            coherence_score = self._calculate_semantic_coherence(current_chunk, sentence)
            
            # Adjust target size based on coherence
            adjusted_target = target_size * (1.0 + coherence_score * 0.2)
            
            should_break = (
                current_tokens + sentence_tokens > adjusted_target and
                len(current_chunk) > 0
            )
            
            if should_break:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
                
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            char_offset += len(sentence) + 1
        
        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_adaptive_chunk(chunk_text, doc_id, len(chunks), char_offset - len(chunk_text)))
        
        return chunks
    
    def _calculate_semantic_coherence(self, current_chunk: List[str], new_sentence: str) -> float:
        """
        Calculate semantic coherence between current chunk and new sentence.
        Returns score from 0.0 (low coherence) to 1.0 (high coherence).
        """
        if not current_chunk:
            return 0.5
        
        # Simple keyword overlap heuristic
        current_text = ' '.join(current_chunk).lower()
        new_text = new_sentence.lower()
        
        current_words = set(re.findall(r'\w+', current_text))
        new_words = set(re.findall(r'\w+', new_text))
        
        if not current_words or not new_words:
            return 0.0
        
        overlap = len(current_words.intersection(new_words))
        total_unique = len(current_words.union(new_words))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _create_adaptive_chunk(self, text: str, doc_id: str, chunk_num: int, start_char: int) -> DocumentChunk:
        """Create an adaptive chunk with enhanced metadata."""
        chunk_id = f"{doc_id}_adaptive_{chunk_num}"
        token_count = self.count_tokens(text)
        
        return DocumentChunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=start_char + len(text),
                token_count=token_count,
                semantic_score=self._calculate_chunk_quality_score(text)
            )
        )
    
    def _calculate_chunk_quality_score(self, text: str) -> float:
        """Calculate a quality score for the chunk (0.0 to 1.0)."""
        # Simple heuristics for chunk quality
        words = text.split()
        if not words:
            return 0.0
        
        # Length score (prefer medium-length chunks)
        length_score = 1.0 - abs(len(words) - 100) / 200.0
        length_score = max(0.0, min(1.0, length_score))
        
        # Completeness score (prefer complete sentences)
        complete_sentences = len(re.findall(r'[.!?]', text))
        total_sentences = len(re.split(r'[.!?]', text))
        completeness_score = complete_sentences / max(total_sentences, 1)
        
        # Combine scores
        return (length_score * 0.6 + completeness_score * 0.4)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]