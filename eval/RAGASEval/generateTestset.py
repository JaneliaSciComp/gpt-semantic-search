
"""WEB LOADER"""

import argparse
import os
import sys
import logging
import warnings
from langchain_community.document_loaders import TextLoader
import glob
import json
import mimetypes
from bs4 import BeautifulSoup
import bs4 as bs
import html2text
import re
from decimal import Decimal
from llama_index.core import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
import pandas as pd
import pyarrow
import random
from langchain_community.embeddings import OllamaEmbeddings




text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.images_to_alt = True
text_maker.single_line_break = True
text_maker.ignore_emphasis = True
SOURCE = "Web"


def webpage_to_text(soup):
    """ Convert a generic web page to searchable text
    """
    title = soup.title.text
    text = text_maker.handle(str(soup))
    return title,text


def janelia_org_to_text(soup):
    """ Convert a janelia.org page to searchable text
    """
    title = soup.title.text.replace(" | Janelia Research Campus","")
    content_sections = soup.find_all("section", class_="content-section")
    if not content_sections:
        return title,None
    if len(content_sections) > 1:
        raise Exception("More than one content section")
    content = content_sections[0]
    # Remove useless content
    for div in content.find_all("div", {'class':['panels-ipe-label','secondary_menu']}):
        div.decompose()
    # Html2text smashes text together if only tags separate it
    # This fix not only adds the spacing but also adds a separator for nav buttons
    for span in content.find_all("span", {'class':'button-wrapper'}):
        sep = bs.NavigableString(" / ")
        span.insert(0, sep)
    text = text_maker.handle(str(content))
    return title,text


def html_to_text(link, body):
    """ Convert a web page to plain text for use as a GPT prompt.
    """
    soup = bs.BeautifulSoup(body,'lxml')
    if "janelia.org" in link:
        title,text = janelia_org_to_text(soup)
    else:
        title,text = webpage_to_text(soup)
    return title,text


class WebSiteLoader():

    def __init__(self, data_path):
        self.data_path = data_path

    def create_document(self, name, title, link, doc_text):
        metadata = {"source": self.data_path, "title": title, "link": link}
        # Debugging: Print doc_text to ensure it's not empty
        return [Document(page_content=doc_text, metadata=metadata)]
    
    def load_all_documents(self):
        documents = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                filepath = os.path.join(root, name)
                with open(filepath) as f:
                    link = f.readline().strip()
                    body = f.read()
                    title, text = html_to_text(link, body)
                    
                    
                    # print(f"Title: {title}")
                    # print(f"Text: {text}")
                    if text:
                        final_text = title + "\n" + text
                        with open('tempTestGen.txt', 'w') as file:
                            file.write(final_text)
                        loader = TextLoader("./tempTestGen.txt")
                        doc = loader.load()
                        documents.append(doc)
        return documents
    




# Open output.txt in write mode




"""ARCHIVED WIKI LOADRER"""


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


class WikiLoader():

    def __init__(self, data_path):
        self.data_path = data_path

    def create_document(self, name, title, link, doc_text):
        metadata = {"source": self.data_path, "title": title, "link": link}
        return [Document(page_content=doc_text, metadata=metadata)]

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
                    # doc = self.create_document(name, title, link, text)
                    # documents.append(doc)
                    if text:
                        final_text = title + "\n" + text
                        with open('tempTestGen.txt', 'w') as file:
                            file.write(final_text)
                        loader = TextLoader("./tempTestGen.txt")
                        doc = loader.load()
                        documents.append(doc)
        return documents
    


"""ARCHIVED SLACK LOADER"""


warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
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


class ArchivedSlackLoader():

    def __init__(self, data_path, debug=False):
        self.data_path = data_path
        self.id2username = {}
        self.id2realname = {}
        self.channel2id = {}
        self.debug = debug

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
        with open(f"{self.data_path}/users.json", 'r') as f:
            users = json.load(f)
            for user in users:
                yield user


    def get_channels(self):
        """ Generator which returns channels from the channels.json file.
        """
        with open(f"{self.data_path}/channels.json", 'r') as f:
            channels = json.load(f)
            for channel in channels:
                yield channel


    def get_messages(self, channel_name):
        """ Generator which returns messages from the json files in the given channel directory.
        """
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
                    # logger.error(f"No such user '{user_id}'")
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
        final_text = doc_text
        with open('tempTestGen.txt', 'w') as file:
            file.write(final_text)
        loader = TextLoader("./tempTestGen.txt")
        
        return loader.load()

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

# data_path = '../data/slack/janelia-software/slack_to_2023-05-18'
# loader = ArchivedSlackLoader(data_path)
# documents = loader.load_all_documents()
# with open('documents.txt', 'w') as file:
#     for document in documents:
#         file.write(str(document) + '\n')


"""DOCUMENT LOADER ALL SOURCES"""


# DONE: Recursively load all scraped files in the directory and its subdirectories
# loader = TextLoader("./test.txt")
# documents = loader.load()
# print(documents)


"""ArchivedSlackLoader
slack_to_2023-05-18

"""
class DocumentLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_documents(self):
        all_documents = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name == "wiki":
                    loader = WikiLoader(folder_path)
                elif folder_name == "slack":
                    # Specify the two subdirectories for slack
                    subdirs = ["janelia-software/slack_to_2023-05-18"]
                    for subdir in subdirs:
                        subfolder_path = os.path.join(folder_path, subdir)
                        # Check if the subdirectory exists
                        if os.path.isdir(subfolder_path):
                            loader = ArchivedSlackLoader(subfolder_path)
                            documents = loader.load_all_documents()
                            # Take a random 10% sample
                            sample_size = max(1, len(documents) * 3 // 100)
                            documents_sample = random.sample(documents, sample_size)
                            all_documents.extend(documents_sample)
                elif folder_name == "janelia.com":
                    loader = WebSiteLoader(folder_path)
                else:
                    continue  # Skip if folder doesn't match any criteria
                # For non-slack directories
                if folder_name != "slack":
                    documents = loader.load_all_documents()
                    # Take a random 10% sample
                    sample_size = max(1, len(documents) * 6 // 100)                    
                    documents_sample = random.sample(documents, sample_size)
                    all_documents.extend(documents_sample)
        return all_documents
    
    def test_load_documents(self):
        # This method is for testing purposes and will only load the first document in each folder path
        all_documents = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name == "wiki":
                    loader = WikiLoader(folder_path)
                elif folder_name == "slack":
                    # Specify the two subdirectories for slack
                    subdirs = ["janelia-software/slack_to_2023-05-18"]
                    for subdir in subdirs:
                        subfolder_path = os.path.join(folder_path, subdir)
                        # Check if the subdirectory exists
                        if os.path.isdir(subfolder_path):
                            loader = ArchivedSlackLoader(subfolder_path)
                            documents = loader.load_all_documents()
                            # Only load the first document for testing
                            if documents:
                                all_documents.append(documents[0])
                elif folder_name == "janelia.com":
                    loader = WebSiteLoader(folder_path)
                else:
                    continue  # Skip if folder doesn't match any criteria
                # For non-slack directories
                if folder_name != "slack":
                    documents = loader.load_all_documents()
                    # Only load the first document for testing
                    if documents:
                        all_documents.append(documents[0])
        return all_documents

# Assuming your data folder is at "./data/"
loader = DocumentLoader("../../data")
documents = loader.test_load_documents()
# print (documents)
# Assuming your data folder is at "../data/"
with open('documents.txt', 'w') as file:
    for document in documents:
        file.write(str(document) + '\n')
# Assuming documents is a list of strings or convertible to string


# Now `final_df` contains all the generated testsets in one DataFrame

base_url_gen = "http://127.0.0.1:11434"
generator_llm = Ollama(model="llama3.1:70b-instruct-q5_1", base_url=base_url_gen)
critic_llm = Ollama(model="llama3.1:70b-instruct-q5_1", base_url=base_url_gen)

base_url_embed = "http://127.0.0.1:11434"
embeddings = OllamaEmbeddings(
    model="avr/sfr-embedding-mistral", 
    base_url=base_url_embed
)

# generator_llm = Ollama(model="llama3")
# critic_llm = Ollama(model="llama3")
# embeddings = OllamaEmbeddings(
#     model="avr/sfr-embedding-mistral"
# )

# generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
# critic_llm = ChatOpenAI(model="gpt-3.5-turbo")


generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
) 




def datasetFix(df):
    columns_to_keep = ['question', 'ground_truth', 'contexts']
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}")
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    df = df[columns_to_keep]
    
    # Apply transformations
    df['contexts'] = df['contexts'].apply(lambda x: [[y] for y in x] if isinstance(x, list) and all(isinstance(y, str) for y in x) else x)
    df['ground_truth'] = df['ground_truth'].apply(lambda x: [[y] for y in x] if isinstance(x, list) and all(isinstance(y, str) for y in x) else x)

    return df

all_data_df = pd.DataFrame()
numiter = 0
total_documents = len(documents)
eighth_documents = total_documents // 8
processed_documents = 0

all_data_df = pd.DataFrame()
for document in documents:

    test_size_gen = int(len(str(document)) // 750)
    logger.info(f"Test size: {test_size_gen}")
    
    # Only generate test cases if test_size_gen is 1 or more
    if test_size_gen >= 1:
        current_testset = generator.generate_with_langchain_docs(document, test_size=test_size_gen, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
        current_testset = current_testset.to_pandas()
        
        
        
        
        # Check if all required columns exist before concatenation
        required_columns = ['question', 'ground_truth', 'contexts']
        if all(column in current_testset.columns for column in required_columns):
            current_df = datasetFix(current_testset)
            all_data_df = pd.concat([all_data_df, current_df], ignore_index=True)
        else:
            logger.warning(f"Skipping document due to missing columns. Required columns: {required_columns}, Found columns: {list(current_testset.columns)}")
    else:
        logger.info("Skipping generation due to test_size_gen < 1")
    processed_documents += 1
    if processed_documents % eighth_documents == 0 or processed_documents == total_documents:
                filename = f'ragasTestSet_{processed_documents // eighth_documents}.parquet'
                all_data_df.to_parquet(filename, index=False)
                logger.info(f"Exported {processed_documents} documents to {filename}")


all_data_df.to_parquet('ragasTestSet.parquet', index=False)