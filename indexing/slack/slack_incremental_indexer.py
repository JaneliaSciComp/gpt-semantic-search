#!/usr/bin/env python3
"""Slack Incremental Indexer - Smart processing of scraped data with built-in discovery."""

import os
import sys
import json
import logging
import argparse
import shutil
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any, Tuple

from llama_index.core import Document

from indexing.weaviate_indexer import Indexer

def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"slack_indexer_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode for daily logs
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

SOURCE = "Slack"
DOCUMENT_PAUSE_SECS = 300
IGNORED_SUBTYPES = set(['channel_join','channel_leave','bot_message'])


def get(dictionary, key):
    """Get the key out of the dictionary, if it exists."""
    if dictionary and key in dictionary:
        return dictionary[key]
    return None


def fix_text(text):
    """Standard transformations on text like squashing multiple newlines."""
    return text.replace("\n\n+", "\n") if text else ""


def find_folders_to_process(data_path: str = "../../data/slack", 
                          indexer: Indexer = None, 
                          buffer_hours: float = 0.5) -> List[Tuple[str, str, float]]:
    """Find all successful folders that need indexing based on database state."""
    if not os.path.exists(data_path):
        return []
    
    # Get latest timestamp from database
    latest_db_timestamp = 0.0
    if indexer:
        try:
            latest_db_timestamp = indexer.get_latest_slack_timestamp()
        except Exception as e:
            logger.warning(f"Could not query database timestamp: {e}")
    
    # Apply buffer
    buffer_seconds = buffer_hours * 3600
    cutoff_timestamp = latest_db_timestamp - buffer_seconds
    
    logger.info(f"Latest database timestamp: {latest_db_timestamp} "
               f"({datetime.fromtimestamp(latest_db_timestamp) if latest_db_timestamp > 0 else 'None'})")
    logger.info(f"Cutoff timestamp (with {buffer_hours}h buffer): {cutoff_timestamp}")
    
    folders_to_process = []
    
    for workspace_dir in os.listdir(data_path):
        workspace_path = os.path.join(data_path, workspace_dir)
        if not os.path.isdir(workspace_path) or workspace_dir == "checkpoint":
            continue
        
        for folder in os.listdir(workspace_path):
            if folder.startswith("success_run_"):
                try:
                    timestamp = float(folder.split("_")[-1])
                    if timestamp > cutoff_timestamp:
                        folders_to_process.append((workspace_dir, folder, timestamp))
                        logger.info(f"Will index: {folder} ({datetime.fromtimestamp(timestamp)})")
                except ValueError:
                    continue
    
    folders_to_process.sort(key=lambda x: x[2])
    logger.info(f"Found {len(folders_to_process)} folders to index")
    
    return folders_to_process


def find_failed_folders(data_path: str = "../../data/slack") -> List[Tuple[str, str, float]]:
    """Find all failed folders for retry."""
    if not os.path.exists(data_path):
        return []
    
    failed_folders = []
    
    for workspace_dir in os.listdir(data_path):
        workspace_path = os.path.join(data_path, workspace_dir)
        if not os.path.isdir(workspace_path) or workspace_dir == "checkpoint":
            continue
        
        for folder in os.listdir(workspace_path):
            if folder.startswith("failed_run_"):
                try:
                    timestamp = float(folder.split("_")[-1])
                    failed_folders.append((workspace_dir, folder, timestamp))
                except ValueError:
                    continue
    
    return sorted(failed_folders, key=lambda x: x[2])


def mark_folder_status(workspace_name: str, folder_name: str, status: str, logger: logging.Logger) -> bool:
    """Mark folder with new status."""
    current_path = Path(f"../../data/slack/{workspace_name}/{folder_name}")
    
    if not current_path.exists():
        logger.error(f"Folder does not exist: {current_path}")
        return False
    
    # Extract timestamp
    try:
        timestamp = folder_name.split("_")[-1]
        int(timestamp)  
    except (ValueError, IndexError):
        logger.error(f"Could not extract timestamp from folder: {folder_name}")
        return False
    
    if status == "success":
        new_folder = f"success_run_{timestamp}"
    elif status == "failed":
        new_folder = f"failed_run_{timestamp}"
    elif status == "processing":
        new_folder = f"run_{timestamp}"
    else:
        logger.error(f"Unknown status: {status}")
        return False
    
    new_path = Path(f"../../data/slack/{workspace_name}/{new_folder}")
    
    if current_path == new_path:
        return True
    
    try:
        current_path.rename(new_path)
        logger.info(f"Renamed folder: {folder_name} -> {new_folder}")
        return True
    except OSError as e:
        logger.error(f"Failed to rename folder: {e}")
        return False


class SlackLoader:
    """Simplified Slack data loader."""

    def __init__(self, data_path: str, debug: bool = False):
        self.data_path = data_path
        self.debug = debug
        self.id2username = {}
        self.id2realname = {}
        self.channel2id = {}

        # Load users and channels
        users_file = os.path.join(data_path, "users.json")
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                for user in json.load(f):
                    user_id = user['id']
                    self.id2username[user_id] = user['name']
                    self.id2realname[user_id] = user['profile']['real_name']

        channels_file = os.path.join(data_path, "channels.json")
        if os.path.exists(channels_file):
            with open(channels_file, 'r') as f:
                for channel in json.load(f):
                    self.channel2id[channel['name']] = channel['id']

    def extract_text(self, elements):
        """Recursively parse an 'elements' structure."""
        text = ''
        for element in elements:
            if 'elements' in element:
                text += self.extract_text(element['elements'])
            el_type = get(element, 'type')
            if el_type == 'text':
                if get(get(element, 'style'), 'code'): text += '`'
                text += element['text']
                if get(get(element, 'style'), 'code'): text += '`'
            elif el_type == 'link':
                text += get(element, 'url')
            elif el_type == 'rich_text_preformatted':
                text += "\n"
            elif el_type == 'user':
                user_id = element['user_id']
                text += self.id2realname.get(user_id, user_id)
        return text

    def parse_message(self, message):
        """Parse a message into text. Returns message timestamp and formatted text."""
        if get(message, 'type') != 'message':
            return None, None
            
        if 'subtype' in message and get(message, 'subtype') in IGNORED_SUBTYPES:
            return None, None
        
        ts = Decimal(message['ts'])
        user_id = message.get('user', 'unknown')

        try:
            realname = self.id2realname[user_id]
        except KeyError:
            try:
                realname = message['user_profile']['display_name']
            except KeyError:
                realname = user_id
            
        if 'blocks' in message:
            text = self.extract_text(message['blocks'])
        else:
            text = message.get('text', '')
        
        # Handle user mentions in text
        text_msg = re.sub(r"<@([^>]+)>", lambda m: self.id2realname.get(m.group(1), m.group(1)), text)
        text_msg = fix_text(text_msg)

        # Handle attachments and files
        if 'attachments' in message:
            for attachment in message['attachments']:
                if 'title' in attachment: text_msg += f"\n{fix_text(attachment['title'])}"
                if 'text' in attachment: text_msg += f"\n{fix_text(attachment['text'])}"
                
        if 'files' in message:
            for file in message['files']:
                if 'name' in file:
                    text_msg += f"\n<{file['name']}>"

        if 'reactions' in message:
            text_msg += f"\nOthers reacted to the previous message with "
            r = [f"{reaction['name']} a total of {reaction['count']} times" for reaction in message['reactions']]
            text_msg += ", and with ".join(r) + "."

        text_msg = f"{realname} said: {text_msg}\n"
        
        return ts, text_msg

    def load_documents(self, channel_name: str) -> List[Document]:
        channel_id = self.channel2id.get(channel_name, channel_name)
        messages = {}  
        
        # Load all JSON files in the channel directory
        channel_dir = os.path.join(self.data_path, channel_name)
        if not os.path.exists(channel_dir):
            return []
        
        for json_file in os.listdir(channel_dir):
            if json_file.endswith('.json'):
                file_path = os.path.join(channel_dir, json_file)
                try:
                    with open(file_path, 'r') as f:
                        for message in json.load(f):
                            ts, text_msg = self.parse_message(message)
                            if ts and text_msg:
                                messages[ts] = text_msg
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue

        if not messages:
            return []

        documents = []
        doc_text = ""
        start_ts = None
        prev_ts = Decimal(0)

        for ts in sorted(messages.keys()):
            # Create a new document whenever messages are separated by a longer pause
            if doc_text and ts - prev_ts > DOCUMENT_PAUSE_SECS:
                doc = Document(text=doc_text, extra_info={
                    "source": SOURCE,
                    "scraped_at": datetime.now().timestamp()
                })
                documents.append(doc)
                doc_text = ""
                start_ts = None

            if not start_ts:
                start_ts = str(ts)
            doc_text += messages[ts]
            prev_ts = ts

        if doc_text:
            doc = Document(text=doc_text, extra_info={
                "source": SOURCE,
                "scraped_at": datetime.now().timestamp()
            })
            documents.append(doc)

        return documents

    def load_all_documents(self) -> List[Document]:
        documents = []
        for channel_name in self.channel2id.keys():
            channel_docs = self.load_documents(channel_name)
            if channel_docs:
                logger.info(f"Found {len(channel_docs)} documents in {channel_name}")
                documents.extend(channel_docs)
        return documents


def process_folder(workspace_name: str, folder_name: str, weaviate_url: str, 
                  class_prefix: str, debug: bool) -> bool:
    """Process a single folder for indexing."""
    folder_path = f"../../data/slack/{workspace_name}/{folder_name}"
    
    try:
        logger.info(f"Processing folder: {folder_name}")
        
        loader = SlackLoader(folder_path, debug=debug)
        documents = loader.load_all_documents()
        logger.info(f"Loaded {len(documents)} documents")

        indexer = Indexer(weaviate_url, class_prefix, False) 
        
        # Log each document being indexed with preview
        for i, doc in enumerate(documents, 1):
            text_preview = doc.text[:20].replace('\n', ' ') if doc.text else ""
            logger.info(f"Indexing document {i}/{len(documents)}: '{text_preview}...' (source: {doc.extra_info.get('source', 'Unknown')})")
        
        indexer.index(documents)

        logger.info(f"Successfully indexed {len(documents)} documents")
        return True
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False


def retry_failed_folders(data_path: str, weaviate_url: str, class_prefix: str, debug: bool) -> Dict[str, int]:
    """Retry all failed folders."""
    failed_folders = find_failed_folders(data_path)
    
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    
    if not failed_folders:
        logger.info("No failed folders found for retry")
        return stats
    
    logger.info(f"Found {len(failed_folders)} failed folders to retry")
    
    for workspace_name, folder_name, timestamp in failed_folders:
        logger.info(f"Retrying failed folder: {folder_name}")
        stats["attempted"] += 1
        
        # Reset to processing state
        if not mark_folder_status(workspace_name, folder_name, "processing", logger):
            stats["failed"] += 1
            continue
        
        processing_folder = f"run_{int(timestamp)}"
        
        # Process the folder
        success = process_folder(workspace_name, processing_folder, weaviate_url, class_prefix, debug)
        
        if success:
            mark_folder_status(workspace_name, processing_folder, "success", logger)
            stats["succeeded"] += 1
            logger.info(f"Successfully retried: {folder_name}")
        else:
            mark_folder_status(workspace_name, processing_folder, "failed", logger)
            stats["failed"] += 1
            logger.error(f"Retry failed: {folder_name}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Slack Incremental Indexer - smart processing of scraped data')
    parser.add_argument('-i', '--input', type=str, help='Path to specific folder to process')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Janelia", help='Class prefix in Weaviate')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--retry-only', action='store_true', help='Only retry failed folders')
    parser.add_argument('--buffer', type=float, default=0.5, help='Buffer time in hours for indexing cutoff')
    parser.add_argument('--data-path', type=str, default="../../data/slack", help='Path to slack data directory')
    
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # First, retry any failed folders
    logger.info("Starting failed folder retry process...")
    retry_stats = retry_failed_folders(args.data_path, args.weaviate_url, args.class_prefix, args.debug)
    
    if retry_stats["attempted"] > 0:
        logger.info(f"Retry summary: {retry_stats['succeeded']} succeeded, {retry_stats['failed']} failed out of {retry_stats['attempted']} attempted")
    
    # If retry-only mode, exit after retrying failed folders
    if args.retry_only:
        logger.info("Retry-only mode: completed retrying failed folders")
        return 0 if retry_stats["failed"] == 0 else 1

    # Smart folder discovery
    if args.input:
        # Manual input mode
        logger.info(f"Manual input mode: processing {args.input}")
        
        # Extract workspace and folder from path
        path_parts = os.path.normpath(args.input).split(os.sep)
        workspace_name = None
        folder_name = None
        
        for i, part in enumerate(path_parts):
            if part == "slack" and i + 1 < len(path_parts):
                workspace_name = path_parts[i + 1]
                if i + 2 < len(path_parts):
                    folder_name = path_parts[i + 2]
                break
        
        if not workspace_name or not folder_name:
            logger.error(f"Could not extract workspace/folder from path: {args.input}")
            return 1
        
        folders_to_process = [(workspace_name, folder_name, 0)]
        
    else:
        # Smart auto-discovery mode
        logger.info("Smart auto-discovery mode: finding folders that need indexing")
        indexer = Indexer(args.weaviate_url, args.class_prefix, False)
        folders_to_process = find_folders_to_process(args.data_path, indexer, args.buffer)
    
    if not folders_to_process:
        logger.info("No folders found that need indexing")
        return 0
    
    logger.info(f"Processing {len(folders_to_process)} folders...")
    
    total_success = 0
    total_failed = 0
    
    for workspace_name, folder_name, timestamp in folders_to_process:
        logger.info(f"Processing folder: {folder_name} ({datetime.fromtimestamp(timestamp) if timestamp > 0 else 'manual'})")
        
        success = process_folder(workspace_name, folder_name, args.weaviate_url, args.class_prefix, args.debug)
        
        if success:
            # Only mark status if it's a success_ folder (from auto-discovery)
            if folder_name.startswith("success_"):
                mark_folder_status(workspace_name, folder_name, "success", logger)
            total_success += 1
        else:
            # Only mark status if it's a success_ folder (from auto-discovery)
            if folder_name.startswith("success_"):
                mark_folder_status(workspace_name, folder_name, "failed", logger)
            total_failed += 1
    
    logger.info(f"Indexing completed: {total_success} succeeded, {total_failed} failed")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    exit(main())