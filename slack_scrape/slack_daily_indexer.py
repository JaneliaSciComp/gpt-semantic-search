#!/usr/bin/env python3
"""Daily Slack Message Indexer - Process scraped messages into Weaviate."""

import os
import sys
import json
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict

try:
    from llama_index.core import Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from weaviate_indexer import Indexer
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False

def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"slack_daily_indexer_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("slack_daily_indexer")


def get_indexing_state() -> Dict[str, float]:
    state_file = os.path.join("logs", "last_indexing_state.json")
    
    if not os.path.exists(state_file):
        return {}
    
    with open(state_file, 'r') as f:
        return json.load(f)


def update_indexing_state(date_str: str, timestamp: float) -> None:
    os.makedirs("logs", exist_ok=True)
    state_file = os.path.join("logs", "last_indexing_state.json")
    
    state = get_indexing_state()
    state[date_str] = timestamp
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def get_messages_for_indexing(workspace_name: str, start_date: str) -> Dict[str, List[Dict[str, Any]]]:
    data_base_path = "../data/slack"
    workspace_path = os.path.join(data_base_path, workspace_name)
    all_messages = defaultdict(list)
    
    if not os.path.exists(workspace_path):
        return all_messages
    
    for date_dir in os.listdir(workspace_path):
        if not date_dir.startswith("slack_to_"):
            continue
            
        date_str = date_dir.replace("slack_to_", "")
        if date_str < start_date:
            continue
        
        date_path = os.path.join(workspace_path, date_dir)
        if not os.path.isdir(date_path):
            continue
        
        for channel_dir in os.listdir(date_path):
            channel_path = os.path.join(date_path, channel_dir)
            if not os.path.isdir(channel_path):
                continue
            
            message_file = os.path.join(channel_path, f"{date_str}.json")
            if os.path.exists(message_file):
                with open(message_file, 'r') as f:
                    messages = json.load(f)
                    all_messages[channel_dir].extend(messages)
    
    return all_messages


def parse_message_for_indexing(message: Dict[str, Any], channel_name: str) -> Tuple[Optional[Decimal], Optional[str]]:
    IGNORED_SUBTYPES = {'channel_join', 'channel_leave', 'bot_message'}
    
    if message.get('type') != 'message':
        return None, None
    
    if 'subtype' in message and message.get('subtype') in IGNORED_SUBTYPES:
        return None, None
    
    ts = message['ts']
    thread_ts = message.get('thread_ts') or ts
    thread_id = Decimal(thread_ts)

    user_id = message.get('user', 'unknown')
    user_profile = message.get('user_profile', {})
    realname = user_profile.get('display_name', user_id)
    
    # Extract text content
    text = ""
    if 'blocks' in message:
        for block in message['blocks']:
            if block.get('type') == 'rich_text':
                for element in block.get('elements', []):
                    for subelement in element.get('elements', []):
                        if subelement.get('type') == 'text':
                            text += subelement.get('text', '')
    else:
        text = message.get('text', '')
    
    text_msg = re.sub(r"\n+", "\n", text)
    
    # Add attachment content
    if 'attachments' in message:
        for attachment in message['attachments']:
            if 'title' in attachment: 
                clean_title = re.sub(r'\n+', '\n', attachment['title'])
                text_msg += f"\n{clean_title}"
            if 'text' in attachment: 
                clean_text = re.sub(r'\n+', '\n', attachment['text'])
                text_msg += f"\n{clean_text}"
    
    # Add file names
    if 'files' in message:
        for file in message['files']:
            if 'name' in file:
                text_msg += f"\n<{file['name']}>"
    
    # Add reactions
    if 'reactions' in message:
        text_msg += f"\nOthers reacted to the previous message with "
        reaction_descriptions = [
            f"{reaction['name']} a total of {reaction['count']} times" 
            for reaction in message['reactions']
        ]
        text_msg += ", and with ".join(reaction_descriptions) + "."

    return thread_id, f"{realname} said: {text_msg}\n"


def process_messages_to_documents(messages: List[Dict[str, Any]], channel_name: str) -> List[Any]:
    if not LLAMA_INDEX_AVAILABLE or not messages:
        return []
    
    DOCUMENT_PAUSE_SECS = 300
    SOURCE = "slack"
    
    # Group messages by thread
    message_threads = {}
    for message in messages:
        thread_id, text_msg = parse_message_for_indexing(message, channel_name)
        if thread_id and text_msg:
            if thread_id not in message_threads:
                message_threads[thread_id] = []
            message_threads[thread_id].append(text_msg)

    if not message_threads:
        return []
    
    # Create documents with pauses between distant threads
    documents = []
    prev_id = Decimal(0)
    doc_text = ""
    start_ts = None

    for thread_id in sorted(message_threads.keys()):
        if doc_text and thread_id - prev_id > DOCUMENT_PAUSE_SECS:
            documents.append(Document(
                text=doc_text, 
                extra_info={
                    "source": SOURCE, 
                    "channel": channel_name, 
                    "ts": start_ts,
                    "title": f"Slack conversation in #{channel_name}",
                    "link": f"https://slack.com/channels/{channel_name}"
                }
            ))
            doc_text = ""
            start_ts = None

        if not start_ts:
            start_ts = str(thread_id)

        doc_text += "".join(message_threads[thread_id])
        prev_id = thread_id

    if doc_text:
        documents.append(Document(
            text=doc_text, 
            extra_info={
                "source": SOURCE, 
                "channel": channel_name, 
                "ts": start_ts,
                "title": f"Slack conversation in #{channel_name}",
                "link": f"https://slack.com/channels/{channel_name}"
            }
        ))

    return documents


def index_documents(documents: List[Any], weaviate_url: str, class_prefix: str) -> bool:
    if not INDEXER_AVAILABLE or not documents:
        return not documents
    
    indexer = Indexer(
        weaviate_url=weaviate_url,
        class_prefix=class_prefix,
        delete_database=False
    )
    
    indexer.index(documents)
    return True


def test_weaviate_connection(weaviate_url: str) -> bool:
    if not INDEXER_AVAILABLE:
        return False
    
    import weaviate
    
    client = weaviate.Client(weaviate_url)
    return client.is_live() and client.is_ready()


def get_document_count(weaviate_url: str, class_prefix: str) -> Optional[int]:
    if not INDEXER_AVAILABLE:
        return None
    
    import weaviate
    
    client = weaviate.Client(weaviate_url)
    class_name = f"{class_prefix}_Node"
    
    result = client.query.aggregate(class_name).with_meta_count().do()
    
    if 'data' in result and 'Aggregate' in result['data']:
        aggregate_data = result['data']['Aggregate']
        if class_name in aggregate_data:
            meta_data = aggregate_data[class_name]
            if meta_data and len(meta_data) > 0 and 'meta' in meta_data[0]:
                return meta_data[0]['meta']['count']
    
    return 0


def main():
    logger = setup_logging()
    
    if not os.getenv("SCRAPING_SLACK_BOT_TOKEN"):
        logger.error("SCRAPING_SLACK_BOT_TOKEN not found in environment variables")
        return 1
    
    enable_indexing = os.getenv("ENABLE_INDEXING", "").lower() == "true"
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not enable_indexing or not openai_api_key:
        logger.info("Indexing disabled or requirements not met")
        return 0
    
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8777")
    class_prefix = os.getenv("CLASS_PREFIX", "Janelia")
    
    if not INDEXER_AVAILABLE:
        logger.error("Indexing components not available")
        return 1
    
    if not test_weaviate_connection(weaviate_url):
        logger.error("Cannot connect to Weaviate")
        return 1
    
    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday.strftime("%Y-%m-%d")
    
    logger.info(f"Indexing messages from {start_date}")
    
    data_base_path = "../data/slack"
    if not os.path.exists(data_base_path):
        logger.warning("No data directory found")
        return 0
    
    workspace_dirs = [d for d in os.listdir(data_base_path) 
                     if os.path.isdir(os.path.join(data_base_path, d))]
    
    if not workspace_dirs:
        logger.warning("No workspace directories found")
        return 0
    
    total_documents = 0
    
    for workspace_name in workspace_dirs:
        messages_by_channel = get_messages_for_indexing(workspace_name, start_date)
        
        if not messages_by_channel:
            continue
        
        all_documents = []
        
        for channel_name, messages in messages_by_channel.items():
            if not messages:
                continue
            
            documents = process_messages_to_documents(messages, channel_name)
            
            if documents:
                all_documents.extend(documents)
                logger.info(f"{channel_name}: {len(documents)} documents")
        
        if all_documents:
            logger.info(f"Indexing {len(all_documents)} documents for {workspace_name}")
            
            if index_documents(all_documents, weaviate_url, class_prefix):
                total_documents += len(all_documents)
                update_indexing_state(start_date, datetime.now().timestamp())
            else:
                logger.error(f"Failed to index documents for {workspace_name}")
                return 1
    
    logger.info(f"Completed - indexed {total_documents} documents")
    
    doc_count = get_document_count(weaviate_url, class_prefix)
    if doc_count is not None:
        logger.info(f"Total documents in database: {doc_count}")
    
    return 0


if __name__ == "__main__":
    exit(main())