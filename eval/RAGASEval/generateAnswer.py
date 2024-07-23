

import os
import re
import sys
import argparse
import logging
import warnings
import pandas as pd
import weaviate
from slack_sdk import WebClient
import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, PromptHelper, GPTVectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI
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
        return client

    

    def get_query_engine(self):
        # Assuming settings like model, class_prefix, etc., are set elsewhere or passed as parameters
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
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
        return query_engine

    def generate_response(self, query):
        query = re.sub("\"", "", query)
        response = self.query_engine.query(query)
        return response.response

# Example usage
if __name__ == "__main__":
    weaviate_url = "http://localhost:8777"
    service = SemanticSearchService(weaviate_url)

    # response = service.generate_response("what is scicomp")
    answers_list = []

    all_data_df = pd.read_parquet('ragasTestSet.parquet')
    questions_list = all_data_df['question'].tolist()
    
    seen = set()
    questions_list = [x for x in questions_list if not (x in seen or seen.add(x))]

    for index, question in enumerate(questions_list):
        try:
            # Use the question as input to get the answer
            answer = service.generate_response(question)
            # Append the answer to the DataFrame directly at the corresponding index
            all_data_df.at[index, 'answer'] = answer
        except Exception as e:
            print(f"Failed to generate or append answer for question at index {index}: {e}")
            # Optionally, append a None or a custom error message to indicate failure
            all_data_df.at[index, 'answer'] = None

    # No need to check if lengths match since we're appending one at a time
    all_data_df.dropna(subset=['answer'], inplace=True)
    all_data_df.dropna(axis=1, how='all', inplace=True)

    print(all_data_df)

    all_data_df.to_parquet('withAnswersTestset.parquet')
