

import os
import re
import sys
import argparse
import logging
import warnings
import textwrap

import weaviate
import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, PromptHelper, GPTVectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI
warnings.simplefilter("ignore", ResourceWarning)

warnings.simplefilter("ignore", ResourceWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#refactor not to use slack for a source
class SemanticSearchService:
    warnings.simplefilter("ignore")
    def __init__(self, weaviate_url):
        self.weaviate_url = weaviate_url
        self.weaviate_client = self.get_weaviate_client()
        self.query_engine = self.get_query_engine()

    def get_weaviate_client(self):
        client = weaviate.Client(self.weaviate_url)
        if not client.is_live():
            raise Exception(f"Weaviate is not live at {self.weaviate_url}")
        
        logger.info("Weaviate client created")
        return client

    

    def get_query_engine(self):
        # Assuming settings like model, class_prefix, etc., are set elsewhere or passed as parameters
        llm = OpenAI(model="gpt-4o", temperature=0)
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        prompt_helper = PromptHelper(4096, 256, 0.1)

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.prompt_helper = prompt_helper

        vector_store = WeaviateVectorStore(weaviate_client=self.weaviate_client, class_prefix="Janelia")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = GPTVectorStoreIndex([], storage_context=storage_context)

        retriever = VectorIndexRetriever(
            index,
            similarity_top_k=5,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
        )

        query_engine = RetrieverQueryEngine.from_args(retriever)
        logger.info("Query engine created")
        return query_engine
    
    def get_unique_nodes(self, nodes):
        docs_ids = set()
        unique_nodes = list()
        for node in nodes:
            if node.node.ref_doc_id not in docs_ids:
                docs_ids.add(node.node.ref_doc_id)
                unique_nodes.append(node)
        return unique_nodes
    
    def escape_text(self, text):
        text = re.sub("<", "&lt;", text)
        text = re.sub(">", "&gt;", text)
        text = re.sub("([_#])", "\\\1", text)
        return text


    def generate_response(self, query):

    # Escape certain characters which the 
        query = re.sub("\"", "", query)

        response = self.query_engine.query(query)

        msg = f"{response.response}\n\nSources:\n\n"
        for node in self.get_unique_nodes(response.source_nodes):
            extra_info = node.node.extra_info
            text = node.node.text

            text = re.sub("\n+", " ", text)
            text = textwrap.shorten(text, width=100, placeholder="...")
            text = self.escape_text(text)

            source = extra_info['source']

        
            msg += f"* {source}: [{extra_info['title']}]({extra_info['link']})\n"

        return msg

# Example usage
if __name__ == "__main__":
    weaviate_url = "http://localhost:8777"
    service = SemanticSearchService(weaviate_url)
    response = service.generate_response("what is scicomp")
    print("\n\nResponse:")
    print(response)
