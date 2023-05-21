#!/usr/bin/env python

import argparse
import os
import re
import sys
import logging
import warnings

import html2text
from llama_index import Document
from weaviate_indexer import Indexer

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE = "wiki"
DOCUMENT_PAUSE_SECS = 300
MAX_INPUT_SIZE = 4096
NUM_OUTPUT = 256
MAX_CHUNK_OVERLAP = 20


text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True

def wiki_to_text(ancestors, title, labels, body):
    body_text = text_maker.handle(body)
    text =  f"Title: {title}\n"
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
        return Document(doc_text, doc_id=name, extra_info={"source": SOURCE, "title": title, "link": link})


    def load_all_documents(self):
        documents = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                filepath = os.path.join(root, name)
                with open(filepath) as f:
                    link = f.readline().rstrip()
                    ancestors = f.readline().rstrip()
                    title = f.readline().rstrip()
                    labels = f.readline().rstrip()
                    body = re.sub('[\n]+', '\n', "".join(f.readlines()))
                    text = wiki_to_text(ancestors, title, labels, body)
                    doc = self.create_document(name, title, link, text)
                    documents.append(doc)
        return documents


def main():
    
    parser = argparse.ArgumentParser(description='Load the given Confluence Wiki export into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to extracted Slack export directory')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8080", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Wiki", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-d', '--delete-database', default=False, action=argparse.BooleanOptionalAction, help='Delete existing "<prefix>_Node" class in Weaviate before starting.')
    args = parser.parse_args()

    # Load the Slack archive from disk and process it into documents
    loader = ArchivedWikiLoader(args.input)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # Index the documents in Weaviate
    indexer = Indexer(args.weaviate_url, args.class_prefix, args.delete_database)
    indexer.index(documents)

if __name__ == '__main__':
    main()
