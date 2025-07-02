#!/usr/bin/env python

import os
import re
import sys
import argparse
import textwrap
import logging
import warnings
import time
import hashlib
from typing import Dict, List
from functools import wraps

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import PromptHelper, GPTVectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

import weaviate
import streamlit as st
from slack_sdk import WebClient

# Modern RAG imports
try:
    from modern_rag_integration import create_janelia_adapter, ModernRAGConfig
    from components.streaming_ui import (
        StreamingStepRenderer, display_mode_selector, 
        create_performance_controls, run_streaming_query
    )
    MODERN_RAG_AVAILABLE = True
    STREAMING_UI_AVAILABLE = True
except ImportError as e:
    MODERN_RAG_AVAILABLE = False
    STREAMING_UI_AVAILABLE = False
    print(f"Modern RAG/Streaming UI not available: {e}")

import asyncio

st.set_page_config(page_title="JaneliaGPT", page_icon="🔍")

from state import init_state
init_state()

warnings.simplefilter("ignore", ResourceWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
EMBED_MODEL_NAME="text-embedding-3-large"
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1
SURVEY_CLASS = "SurveyResponses"

SIDEBAR_DESC = """
JaneliaGPT uses OpenAI models to index various data sources in a vector database for searching. 
Currently the following sources are indexed:
* Janelia.org
* Janelia-Software Slack Workspace
* Janelia Wiki (spaces 'SCSW', 'SCS', and 'ScientificComputing')
"""

NODE_SCHEMA: List[Dict] = [
    {
        "dataType": ["text"],
        "description": "User query",
        "name": "query"
    },
    {
        "dataType": ["text"],
        "description": "GPT response",
        "name": "response"
    },
    {
        "dataType": ["text"],
        "description": "Survey response",
        "name": "survey",
    },
]

def create_survey_schema(weaviate_client) -> None:
    """Create schema."""
    # first check if schema exists
    schema = weaviate_client.schema.get()
    classes = schema["classes"]
    existing_class_names = {c["class"] for c in classes}
    # if schema already exists, don't create
    if SURVEY_CLASS in existing_class_names:
        return

    properties = NODE_SCHEMA
    class_obj = {
        "class": SURVEY_CLASS,  # <= note the capital "A".
        "description": f"Class for survey responses",
        "properties": properties,
    }
    weaviate_client.schema.create_class(class_obj)


def record_log(weaviate_client, query, response):
    metadata = {
        "query": query,
        "response": response,
        'survey': 'Unknown'
    }
    return weaviate_client.data_object.create(metadata, SURVEY_CLASS)


def record_survey(weaviate_client, db_id, survey):
    metadata = {
        "survey": survey,
    }
    weaviate_client.data_object.update(metadata, SURVEY_CLASS, db_id)


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


@st.cache_data
def get_message_link(_slack_client, channel, ts):
    res = _slack_client.chat_getPermalink(channel=channel, message_ts=ts)
    if res['ok']:
        return res['permalink']
    

@st.cache_resource
def get_weaviate_client(weaviate_url):

    client = weaviate.Client(weaviate_url)

    if not client.is_live():
        raise Exception(f"Weaviate is not live at {weaviate_url}")

    return client


@st.cache_resource
def get_slack_client():
    slack_client = WebClient(token=os.environ.get('SCRAPING_SLACK_USER_TOKEN'))
    res = slack_client.api_test()
    if not res["ok"]:
        logger.error(f"Error initializing Slack API: {res['error']}")
        sys.exit(1)

    return slack_client


def get_query_engine(_weaviate_client):

    model = st.session_state["model"]
    class_prefix = st.session_state["class_prefix"]
    temperature = st.session_state["temperature"] / 100.0
    search_alpha = st.session_state["search_alpha"] / 100.0
    num_results = st.session_state["num_results"]
    
    # Check if modern RAG is enabled
    use_modern_rag = st.session_state.get("use_modern_rag", False) and MODERN_RAG_AVAILABLE

    logger.info("Getting query engine with parameters:")
    logger.info(f"  model: {model}")
    logger.info(f"  class_prefix: {class_prefix}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  search_alpha: {search_alpha}")
    logger.info(f"  num_results: {num_results}")
    logger.info(f"  use_modern_rag: {use_modern_rag}")

    if use_modern_rag:
        # Use modern RAG system
        try:
            config = ModernRAGConfig(
                weaviate_url=args.weaviate_url,
                class_name=f"{class_prefix}_Node",
                llm_model=model,
                temperature=temperature,
                final_top_k=num_results,
                enable_enhanced_retrieval=st.session_state.get("enable_enhanced_retrieval", True),
                enable_hyde=st.session_state.get("enable_hyde", True),
                enable_query_routing=st.session_state.get("enable_query_routing", True),
                enable_reranking=st.session_state.get("enable_reranking", True),
                # Performance optimization settings
                agentic_processing_mode=st.session_state.get("agentic_processing_mode", "balanced"),
                enable_streaming=st.session_state.get("enable_agentic_streaming", True),
                skip_complex_routing=st.session_state.get("skip_complex_routing", False),
                hyde_timeout_seconds=st.session_state.get("hyde_timeout_seconds", 3.0),
                routing_timeout_seconds=st.session_state.get("routing_timeout_seconds", 2.0),
                enable_parallel_processing=st.session_state.get("enable_parallel_processing", True)
            )
            
            adapter = create_janelia_adapter(
                weaviate_url=args.weaviate_url,
                class_prefix=class_prefix,
                enable_modern=True
            )
            adapter.modern_rag.config = config
            
            query_engine = adapter.get_query_engine()
            logger.info("Using modern RAG query engine")
            return query_engine
            
        except Exception as e:
            logger.warning(f"Modern RAG failed, falling back to legacy: {e}")
            use_modern_rag = False

    # Legacy system (original code)
    llm = OpenAI(model=model, temperature=temperature)
    embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
    prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.prompt_helper = prompt_helper

    vector_store = WeaviateVectorStore(weaviate_client=_weaviate_client, class_prefix=class_prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex([], storage_context=storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index,
        similarity_top_k=num_results,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        alpha=search_alpha,
    )

    # construct query engine
    query_engine = RetrieverQueryEngine.from_args(retriever)
    logger.info("Using legacy query engine")

    return query_engine


# Cache for query responses
query_cache = {}

def cache_key(query, model, num_results, temperature, search_alpha):
    """Generate cache key for query parameters"""
    key_str = f"{query}_{model}_{num_results}_{temperature}_{search_alpha}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_response(_query_engine, _slack_client, query):
    
    # Check cache first if enabled
    if st.session_state.get("enable_query_caching", True):
        cache_key_str = cache_key(
            query, 
            st.session_state["model"],
            st.session_state["num_results"],
            st.session_state["temperature"],
            st.session_state["search_alpha"]
        )
        
        if cache_key_str in query_cache:
            if st.session_state.get("debug_mode", False):
                st.info("🎯 Using cached result")
            return query_cache[cache_key_str]
    
    # Show progress if enabled
    show_progress = st.session_state.get("show_progress", True)
    debug_mode = st.session_state.get("debug_mode", False)
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing query...")
    
    start_time = time.time()

    # Escape certain characters which the 
    query = re.sub("\"", "", query)
    
    if show_progress:
        progress_bar.progress(20)
        status_text.text("Generating embeddings...")
    
    try:
        if show_progress:
            progress_bar.progress(40)
            status_text.text("Searching knowledge base...")
        
        response = _query_engine.query(query)
        
        if show_progress:
            progress_bar.progress(80)
            status_text.text("Formatting response...")
            
    except Exception as e:
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        raise e

    msg = f"{response.response}\n\nSources:\n\n"
    for node in get_unique_nodes(response.source_nodes):
        extra_info = node.node.extra_info
        text = node.node.text

        text = re.sub("\n+", " ", text)
        text = textwrap.shorten(text, width=100, placeholder="...")
        text = escape_text(text)

        source = extra_info['source']

        if source.lower() == 'slack':
            channel_id = extra_info['channel']
            ts = extra_info['ts']
            msg += f"* {source}: [{text}]({get_message_link(_slack_client, channel_id, ts)})\n"
        else:
            msg += f"* {source}: [{extra_info['title']}]({extra_info['link']})\n"
    
    # Update progress and cache result
    if show_progress:
        progress_bar.progress(100)
        status_text.text("Query completed!")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
    
    # Cache the result if caching is enabled
    if st.session_state.get("enable_query_caching", True):
        query_cache[cache_key_str] = msg
    
    # Show debug info if enabled
    if debug_mode:
        elapsed_time = time.time() - start_time
        st.info(f"🕰️ Query completed in {elapsed_time:.2f} seconds")
        st.info(f"📋 Found {len(response.source_nodes)} source documents")

    return msg

parser = argparse.ArgumentParser(description='Web service for semantic search using Weaviate and OpenAI')
parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", help='Weaviate database URL')
args = parser.parse_args()

weaviate_client = get_weaviate_client(args.weaviate_url)

st.sidebar.markdown(SIDEBAR_DESC)
st.title("Ask JaneliaGPT")

# Quick Performance Controls
with st.expander("⚡ Quick Performance Controls", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Query Controls**")
        if st.button("Stop Long Queries"):
            st.warning("Query interruption requested. Refresh page if needed.")
        
    with col2:
        cache_enabled = st.checkbox(
            "Enable Caching",
            value=st.session_state.get("enable_query_caching", True),
            key="cache_quick"
        )
        st.session_state.enable_query_caching = cache_enabled
        
    with col3:
        show_debug = st.checkbox(
            "Show Debug Info",
            value=st.session_state.get("debug_mode", False),
            key="debug_quick"
        )
        st.session_state.debug_mode = show_debug
        
    # Cache management
    if cache_enabled and query_cache:
        st.text(f"📊 Cache: {len(query_cache)} queries cached")
        if st.button("Clear Cache"):
            query_cache.clear()
            st.success("Cache cleared!")

# Performance controls for modern RAG (only show if available and enabled)
use_modern_rag = st.session_state.get("use_modern_rag", False) and MODERN_RAG_AVAILABLE
if use_modern_rag and STREAMING_UI_AVAILABLE:
    with st.expander("🔬 Modern RAG Controls", expanded=False):
        processing_mode = display_mode_selector()
        st.session_state.agentic_processing_mode = processing_mode
        
        # Additional streaming controls
        col1, col2 = st.columns(2)
        with col1:
            enable_streaming = st.checkbox(
                "Show Processing Steps", 
                value=st.session_state.get("enable_agentic_streaming", True),
                key="enable_agentic_streaming"
            )
        with col2:
            skip_complex = st.checkbox(
                "Fast Mode (Skip Complex Analysis)",
                value=st.session_state.get("skip_complex_routing", False),
                key="skip_complex_routing"
            )

query = st.text_input("What would you like to ask?", '', key="query")


is_new_query = query and query != st.session_state.last_processed_query

if is_new_query or st.button("Submit"):
    if query:  
        logger.info(f"Query: {query}")
        try:
            query_engine = get_query_engine(weaviate_client)
            slack_client = get_slack_client()
            
            # Check if we should use streaming modern RAG
            use_streaming = (
                use_modern_rag and 
                STREAMING_UI_AVAILABLE and 
                st.session_state.get("enable_agentic_streaming", True)
            )
            
            if use_streaming:
                # Use streaming interface
                step_renderer = StreamingStepRenderer()
                step_renderer.initialize_display()
                
                # Run streaming query
                try:
                    response_data = asyncio.run(run_streaming_query(query_engine, query, step_renderer))
                    if response_data and "response" in response_data:
                        msg = response_data["response"]
                        # Add sources from response_data if available
                        if "sources" in response_data:
                            msg += "\n\nSources:\n\n"
                            for source in response_data["sources"]:
                                msg += f"* {source}\n"
                    else:
                        # Fallback to regular processing
                        msg = get_response(query_engine, slack_client, query)
                    
                    step_renderer.render_summary()
                    
                except Exception as streaming_error:
                    st.warning(f"Streaming failed: {streaming_error}. Falling back to standard processing.")
                    msg = get_response(query_engine, slack_client, query)
            else:
                # Use standard response function
                msg = get_response(query_engine, slack_client, query)
            
            # Only create a new log entry if this is truly a new query
            if query != st.session_state.last_processed_query:
                st.session_state.db_id = record_log(weaviate_client, query, msg)
                st.session_state.last_processed_query = query
                st.session_state.survey_complete = False
            
            st.session_state.response = msg
            st.session_state.response_error = False
            logger.info(f"Response saved as {st.session_state.db_id}: {msg}")
            st.success(msg)
        except Exception as e:
            msg = f"An error occurred: {e}"
            st.session_state.response = msg
            st.session_state.response_error = True
            st.session_state.last_processed_query = query
            logger.exception(msg)
            st.error(msg)

elif st.session_state.response:
    if st.session_state.response_error:
        st.error(st.session_state.response)
    else:
        st.success(st.session_state.response)


def survey_click(survey_response):

    st.session_state.survey = survey_response
    st.session_state.survey_complete = True

    create_survey_schema(weaviate_client)

    db_id = st.session_state.db_id
    record_survey(weaviate_client, db_id, survey_response)
    logger.info(f"Logged survey response: {survey_response}")
    del st.session_state['survey']


if st.session_state.response and not st.session_state.survey_complete:
    st.markdown(
        """
        <style>
            div[data-testid="column"]:nth-of-type(1)
            {
                text-align: end;
            } 
        </style>
        """,unsafe_allow_html=True
    )

    with st.form(key="survey_form"):
        st.markdown("Was your question answered?")
        col1, col2 = st.columns([1,1])
        with col1:
            st.form_submit_button("Yes", on_click=survey_click, args=('Yes', ))
        with col2:
            st.form_submit_button("No", on_click=survey_click, args=('No', ))