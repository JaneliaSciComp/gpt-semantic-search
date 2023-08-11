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
SOURCE = "Wiki"

text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True

def wiki_to_text(ancestors, title, labels, body):
    """ Convert a wiki document to plain text for use as a GPT prompt.
    """
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
        return Document(text=doc_text, doc_id=name, extra_info={"source": SOURCE, "title": title, "link": link})

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
    parser.add_argument('-r', '--remove-existing', default=False, action=argparse.BooleanOptionalAction, help='Remove existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction, help='Print debugging information, such as the message content.')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load the Slack archive from disk and process it into documents
    loader = ArchivedWikiLoader(args.input)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # Index the documents in Weaviate
    indexer = Indexer(args.weaviate_url, args.class_prefix, args.remove_existing)
    indexer.index(documents)

if __name__ == '__main__':
    main()
