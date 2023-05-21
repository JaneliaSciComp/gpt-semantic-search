#!/usr/bin/env python

import re
import textwrap
import streamlit as st

from llama_index import LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index import ResponseSynthesizer
from llama_index.query_engine import RetrieverQueryEngine

# Debug logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)

import weaviate
# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

#gpt-3.5-turbo
llm = ChatOpenAI(temperature=0.9, model_name="text-curie-001")
llm_predictor = LLMPredictor(llm=llm)
embed_model = LangchainEmbedding(OpenAIEmbeddings())

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)

class_prefix = "Slack"
vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex([], storage_context=storage_context)

import os
from slack_sdk import WebClient
client = WebClient(token=os.environ.get('SLACK_TOKEN'))
res = client.api_test()
if not res["ok"]:
    raise ValueError(f"Error initializing Slack API: {res['error']}")

def get_message_link(channel, ts):
    res = client.chat_getPermalink(channel=channel, message_ts=ts)
    if res['ok']:
        return res['permalink']
    
def get_unique_nodes(nodes):
    docs_ids = set()
    unique_nodes = list()
    for node in nodes:
        if node.node.ref_doc_id not in docs_ids:
            docs_ids.add(node.node.ref_doc_id)
            unique_nodes.append(node)
    return unique_nodes

def escape_text(text):
    text = re.sub("<", "&lt;", text)
    text = re.sub(">", "&gt;", text)
    text = re.sub("([_#])", "\\\1", text)
    return text

# configure retriever
retriever = VectorIndexRetriever(
    index,
    similarity_top_k=3,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    alpha=0.5,
)

# configure response synthesizer
synth = ResponseSynthesizer.from_args()

# construct query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synth,
)

st.title("Ask JaneliaGPT")
query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            response = query_engine.query(query)
            msg = f"{response.response}\n\nSources:\n\n"
            for node in get_unique_nodes(response.source_nodes):
                extra_info = node.node.extra_info
                text = node.node.text
                
                text = re.sub("\n+", " ", text)
                text = textwrap.shorten(text, width=80, placeholder="...")
                text = escape_text(text)
                
                source = extra_info['source']

                if source == 'slack':
                    channel_id = extra_info['channel']
                    ts = extra_info['ts']
                    msg += f"* [{text}]({get_message_link(channel_id, ts)})\n"

                elif source == 'wiki':
                    msg += f"* [{extra_info['title']}]({extra_info['link']})\n"

            st.success(msg)

        except Exception as e:
            st.error(f"An error occurred: {e}")

