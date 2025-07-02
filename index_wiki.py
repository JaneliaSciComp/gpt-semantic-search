#!/usr/bin/env python

import argparse
import os
import re
import sys
import logging
import warnings

import html2text
from llama_index.core import Document
#TODO read up ^ (if its chunking large docs automatically)
from weaviate_indexer import Indexer

# Modern RAG support
try:
    from modern_rag_integration import create_modern_rag_system, ModernRAGConfig
    from chunking.advanced_chunkers import ContentType
    MODERN_RAG_AVAILABLE = True
except ImportError:
    MODERN_RAG_AVAILABLE = False
    logger.warning("Modern RAG not available - install modern components for enhanced features")

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE = "Wiki"

text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True


def wiki_to_text(ancestors, title, authors, labels, body):
    """ Convert a wiki document to plain text for use as a GPT prompt.
    """
    body_text = text_maker.handle(body)
    text =  f"Title: {title}\n"
    if authors: text += f"Authors: {authors}\n" 
    if ancestors: text += f"Ancestors: {ancestors}\n" 
    if labels: text += f"Labels: {ancestors}\n"
    text += f"{body_text}"
    return text


class ArchivedWikiLoader():

    def __init__(self, data_path):
        self.data_path = data_path

    def create_document(self, name, title, link, doc_text):
        logger.info(f"Document[name={name},link={link}]")
        logger.debug(doc_text)
        return Document(text=doc_text, doc_id=name, extra_info={"source": SOURCE, "title": title, "link": link})

    def load_all_documents(self):
        documents = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                filepath = os.path.join(root, name)
                with open(filepath) as f:
                    link = f.readline().rstrip()
                    ancestors = f.readline().rstrip()
                    title = f.readline().rstrip()
                    authors = f.readline().rstrip()
                    labels = f.readline().rstrip()
                    body = re.sub('[\n]+', '\n', "".join(f.readlines()))
                    text = wiki_to_text(ancestors, title, authors, labels, body)
                    doc = self.create_document(name, title, link, text)
                    documents.append(doc)
        return documents


def main():
    
    parser = argparse.ArgumentParser(description='Load the given Confluence Wiki export into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to extracted Slack export directory')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Wiki", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-r', '--remove-existing', default=False, action=argparse.BooleanOptionalAction, help='Remove existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction, help='Print debugging information, such as the message content.')
    parser.add_argument('--modern-rag', default=False, action=argparse.BooleanOptionalAction, help='Use modern RAG features (advanced chunking, multi-vector embeddings)')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load the wiki documents from disk and process them
    loader = ArchivedWikiLoader(args.input)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # Index the documents in Weaviate
    if args.modern_rag and MODERN_RAG_AVAILABLE:
        # Use modern RAG system
        logger.info("Using modern RAG indexing with advanced chunking")
        config = ModernRAGConfig(
            weaviate_url=args.weaviate_url,
            class_name=f"{args.class_prefix}_Node",
            chunking_strategy="hierarchical",
            enable_multi_vector=True
        )
        
        modern_rag = create_modern_rag_system(**config.__dict__)
        
        # Create schema if removing existing
        if args.remove_existing:
            modern_rag.create_schema(delete_existing=True)
        else:
            modern_rag.create_schema(delete_existing=False)
        
        # Index with wiki content type for optimal chunking
        modern_rag.index_documents(documents, content_type=ContentType.MARKDOWN)
        
    else:
        # Use legacy indexer
        if args.modern_rag and not MODERN_RAG_AVAILABLE:
            logger.warning("Modern RAG requested but not available, falling back to legacy")
            
        indexer = Indexer(args.weaviate_url, args.class_prefix, args.remove_existing)
        indexer.index(documents, content_type="wiki")

if __name__ == '__main__':
    main()
