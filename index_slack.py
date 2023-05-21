#!/usr/bin/env python

import argparse
import re
import sys
import glob
import json
import logging
import warnings

from decimal import Decimal

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from llama_index import Document, LLMPredictor, PromptHelper, ServiceContext, LangchainEmbedding, GPTVectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext

import weaviate

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE = "slack"
DOCUMENT_PAUSE_SECS = 300
MAX_INPUT_SIZE = 4096
NUM_OUTPUT = 256
MAX_CHUNK_OVERLAP = 20
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


class ArchivedSlackLoader():

    def __init__(self, data_path):
        self.data_path = data_path
        self.id2username = {}
        self.id2realname = {}
        self.channel2id = {}

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
        with open(f"{self.data_path}/users.json", 'r') as f:
            users = json.load(f)
            for user in users:
                yield user


    def get_channels(self):
        with open(f"{self.data_path}/channels.json", 'r') as f:
            channels = json.load(f)
            for channel in channels:
                yield channel


    def get_messages(self, channel_name):
        for messages_file in glob.glob(f"{self.data_path}/{channel_name}/*.json"):
            with open(messages_file, 'r') as f:
                for message in json.load(f):
                    yield message


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

                # Translate user
                user_id = message['user']
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
                    text = message['text']
                
                text_msg = re.sub("<@(.*?)>", lambda m: self.id2realname[m.group(1)], text)
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
        logger.info(doc_text)
        return Document(doc_text, extra_info={"source": SOURCE, "channel": channel_id, "ts": ts})


    def load_documents(self, channel_name):
        channel_id = self.channel2id[channel_name]
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
        doc = self.create_document(channel_id, start_ts, doc_text)
        documents.append(doc)

        return documents


    def load_all_documents(self):
        documents = []
        for channel_name in self.channel2id.keys():
            for doc in self.load_documents(channel_name):
                documents.append(doc)
        return documents


def main():
    
    parser = argparse.ArgumentParser(description='Load the given Slack export into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to extracted Slack export directory')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8080", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Slack", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-d', '--delete-database', default=False, action=argparse.BooleanOptionalAction, help='Delete existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-s', '--storage-dir', type=str, default="./storage/index/slack", help='Path to the index storage directory.')
    args = parser.parse_args()

    # Load the Slack archive from disk and process it into documents
    loader = ArchivedSlackLoader(args.input)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # Connect to Weaviate database
    client = weaviate.Client(args.weaviate_url)

    if not client.is_live():
        logger.error(f"Weaviate is not live at {args.weaviate_url}")
        sys.exit()

    if not client.is_live():
        logger.error(f"Weaviate is not ready at {args.weaviate_url}")
        sys.exit()

    logger.info(f"Connected to Weaviate at {args.weaviate_url} (Version {client.get_meta()['version']})")

    # Delete existing data in Weaviate
    class_prefix = args.class_prefix
    if args.delete_database:
        class_name = f"{class_prefix}_Node"
        logger.warning(f"Deleting {class_name} class in Weaviate")
        client.schema.delete_class(class_name)

    # Create LLM embedding model
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0301")
    llm_predictor = LLMPredictor(llm=llm)
    embed_model = LangchainEmbedding(OpenAIEmbeddings())
    prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, MAX_CHUNK_OVERLAP)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, prompt_helper=prompt_helper)

    # Embed the documents and persist the embeddings into Weaviate    
    logger.info("Creating GPT vector store index")
    vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
    index

    # Persist the docstore and index_store
    # This is currently required although in theory Weaviate should be able to handle these as well
    #logger.info(f"Persisting index to {args.storage_dir}")
    #storage_context.persist(persist_dir=args.storage_dir)

    logger.info(f"Completed indexing {args.input} into '{class_prefix}_Node'")


if __name__ == '__main__':
    main()