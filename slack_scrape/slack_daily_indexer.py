#!/usr/bin/env python

import argparse
import re
import sys
import os
import glob
import json
import logging
import warnings
from datetime import datetime, timedelta
from decimal import Decimal

from llama_index.core import Document

# Add parent directory to path to import weaviate_indexer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from weaviate_indexer import Indexer

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - exactly same as index_slack.py
SOURCE = "Slack"
DOCUMENT_PAUSE_SECS = 300
IGNORED_SUBTYPES = set(['channel_join','channel_leave','bot_message'])


def get(dictionary, key):
    """ Get the key out of the dictionary, if it exists. If not, return None.
    """
    if dictionary and key in dictionary:
        return dictionary[key]
    return None


def fix_text(text):
    """ Standard transformations on text like squashing multiple newlines.
    """
    text = re.sub("\n+", "\n", text)
    return text


class DailySlackLoader():
    """Modified version of ArchivedSlackLoader for daily scraped data."""

    def __init__(self, data_path, target_date=None, debug=False):
        self.data_path = data_path
        self.target_date = target_date or datetime.now().strftime("%Y-%m-%d")
        self.id2username = {}
        self.id2realname = {}
        self.channel2id = {}
        self.debug = debug

        # Load users.json and channels.json (created by slack_daily_scraper)
        for user in self.get_users():
            id = user['id']
            self.id2username[id] = user['name']
            self.id2realname[id] = user['profile']['real_name']

        logger.info(f"Loaded {len(self.id2username)} users")
        for channel in self.get_channels():
            logger.debug(f"{channel['id']}: {channel['name']}")
            self.channel2id[channel['name']] = channel['id']
        
        logger.info(f"Loaded {len(self.channel2id)} channels")


    def get_users(self):
        """ Generator which returns users from the users.json file.
        """
        users_file = f"{self.data_path}/users.json"
        if not os.path.exists(users_file):
            logger.warning(f"No users.json found at {users_file}")
            return
            
        with open(users_file, 'r') as f:
            users = json.load(f)
            for user in users:
                yield user


    def get_channels(self):
        """ Generator which returns channels from the channels.json file.
        """
        channels_file = f"{self.data_path}/channels.json"
        if not os.path.exists(channels_file):
            logger.warning(f"No channels.json found at {channels_file}")
            return
            
        with open(channels_file, 'r') as f:
            channels = json.load(f)
            for channel in channels:
                yield channel


    def get_messages(self, channel_name):
        """ Generator which returns messages from the json files in the given channel directory.
        Only processes messages from the target date.
        """
        # Look for the specific date file in the channel directory
        target_file = f"{self.data_path}/{channel_name}/{self.target_date}.json"
        
        if os.path.exists(target_file):
            logger.debug(f"Processing messages from {target_file}")
            with open(target_file, 'r') as f:
                for message in json.load(f):
                    yield message
        else:
            logger.debug(f"No messages file found for {channel_name} on {self.target_date}")


    def extract_text(self, elements):
        """ Recursively parse an 'elements' structure, 
            converting user elements to their real names.
        """
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
                try:
                    text += self.id2realname[user_id]
                except KeyError:
                    logger.error(f"No such user '{user_id}'")
                    text += user_id

        return text

    def parse_message(self, message):
        """ Parse a message into text that will be read by a GPT model. 
        """
        thread_id, text_msg = None, None
        if get(message, 'type') == 'message':
            if 'subtype' in message and get(message, 'subtype') in IGNORED_SUBTYPES:
                pass
            else:
                ts = message['ts']
                thread_ts = get(message, 'thread_ts') or ts
                thread_id = Decimal(thread_ts)

                # Translate user - handle both formats (scraped vs export)
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
                if user_id in self.id2realname:
                    text_msg = re.sub("<@(.*?)>", lambda m: self.id2realname.get(m.group(1), m.group(1)), text)
                else:
                    text_msg = text
                text_msg = fix_text(text_msg)

                if 'attachments' in message:
                    for attachment in message['attachments']:
                        if 'title' in attachment: text_msg += f"\n{fix_text(attachment['title'])}"
                        if 'text' in attachment: text_msg += f"\n{fix_text(attachment['text'])}"
                        
                if 'files' in message:
                    for file in message['files']:
                        if 'name' in file:
                            # There are several cases where a file doesn't have a name:
                            # 1) The file has been deleted (mode=tombstone)
                            # 2) We have no access (file_access=access_denied)
                            text_msg += f"\n<{file['name']}>"

                if 'reactions' in message:
                    text_msg += f"\nOthers reacted to the previous message with "
                    r = [f"{reaction['name']} a total of {reaction['count']} times" for reaction in message['reactions']]
                    text_msg += ", and with ".join(r) + "."

                text_msg = f"{realname} said: {text_msg}\n"
        
        return thread_id, text_msg


    def create_document(self, channel_id, ts, doc_text):
        logger.info("--------------------------------------------------")
        logger.info(f"Document[channel={channel_id},ts={ts}]")
        logger.debug(doc_text)
        return Document(text=doc_text, extra_info={"source": SOURCE, "channel": channel_id, "ts": ts})


    def load_documents(self, channel_name):
        channel_id = self.channel2id.get(channel_name, channel_name)  # Fallback to channel_name if not found
        messages = {}
        for message in self.get_messages(channel_name):
            try:
                thread_id, text_msg = self.parse_message(message)
            except Exception as e:
                logger.error(f"Error parsing message: {message}")
                raise e
                
            if thread_id and text_msg:
                if thread_id not in messages:
                    messages[thread_id] = []
                messages[thread_id].append(text_msg)

        if not messages:
            logger.debug(f"No messages found for {channel_name} on {self.target_date}")
            return []

        prev_id = Decimal(0)
        documents = []
        doc_text = ""
        start_ts = None

        for thread_id in sorted(list(messages.keys())):

            # Create a new document whenever messages are separated by a longer pause
            if doc_text and thread_id-prev_id > DOCUMENT_PAUSE_SECS:
                doc = self.create_document(channel_id, start_ts, doc_text)
                documents.append(doc)
                doc_text = ""
                start_ts = None

            logger.debug(thread_id)

            # Starting timestamp for the next document
            if not start_ts:
                start_ts = str(thread_id)

            # Add all messages from the current thread
            for text_msg in messages[thread_id]:
                doc_text += text_msg

            prev_id = thread_id

        # Add final document
        if doc_text:  # Only add if there's content
            doc = self.create_document(channel_id, start_ts, doc_text)
            documents.append(doc)

        return documents


    def load_all_documents(self):
        documents = []
        for channel_name in self.channel2id.keys():
            channel_docs = self.load_documents(channel_name)
            if channel_docs:
                logger.info(f"Found {len(channel_docs)} documents in {channel_name} for {self.target_date}")
                documents.extend(channel_docs)
        return documents


def find_export_directory_for_date(target_date, data_base_path="data/slack"):
    """Find the export directory for a specific date."""
    if not os.path.exists(data_base_path):
        return None
    
    target_dir_name = f"slack_to_{target_date}"
    
    for workspace_dir in os.listdir(data_base_path):
        workspace_path = os.path.join(data_base_path, workspace_dir)
        if not os.path.isdir(workspace_path):
            continue
            
        export_path = os.path.join(workspace_path, target_dir_name)
        
        # Check if this directory exists and has required metadata files
        if (os.path.exists(export_path) and
            os.path.exists(os.path.join(export_path, "users.json")) and 
            os.path.exists(os.path.join(export_path, "channels.json"))):
            return export_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Daily Slack indexer - processes scraped Slack data for a specific date into Weaviate')
    parser.add_argument('-i', '--input', type=str, help='Path to extracted Slack export directory (optional, will auto-discover if not provided)')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Janelia", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-r', '--remove-existing', default=False, action=argparse.BooleanOptionalAction, help='Remove existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction, help='Print debugging information, such as the message content.')
    parser.add_argument('--date', type=str, help='Target date to process (YYYY-MM-DD format, defaults to today)')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Determine target date
    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Processing Slack messages for date: {target_date}")

    # Find or use export directory
    if args.input:
        export_dir = args.input
    else:
        export_dir = find_export_directory_for_date(target_date)
    
    if not export_dir:
        logger.error(f"No valid Slack export directory found for {target_date}. "
                    f"Use -i to specify manually or ensure slack_daily_scraper has run for that date.")
        sys.exit(1)
    
    logger.info(f"Using export directory: {export_dir}")
    
    # Load the Slack archive from disk and process it into documents
    loader = DailySlackLoader(export_dir, target_date=target_date, debug=args.debug)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents for {target_date}")

    if not documents:
        logger.info(f"No documents to index for {target_date}")
        sys.exit(0)

    # Index the documents in Weaviate
    indexer = Indexer(args.weaviate_url, args.class_prefix, args.remove_existing)
    indexer.index(documents)

    logger.info(f"Successfully indexed {len(documents)} documents for {target_date}")


if __name__ == '__main__':
    main()