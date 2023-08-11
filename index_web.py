#!/usr/bin/env python

import argparse
import os
import sys
import logging
import warnings

import bs4 as bs
import html2text
from llama_index import Document
from weaviate_indexer import Indexer

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOURCE = "Web"

text_maker = html2text.HTML2Text()
text_maker.ignore_links = True
text_maker.images_to_alt = True
text_maker.single_line_break = True
text_maker.ignore_emphasis = True


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


class ArchivedWebSiteLoader():

    def __init__(self, data_path):
        self.data_path = data_path

    def create_document(self, name, title, link, doc_text):
        logger.info(f"Document[id={name},title={title},link={link}]")
        logger.debug(doc_text)
        return Document(text=doc_text, doc_id=name, extra_info={"source": SOURCE, "title": title, "link": link})

    def load_all_documents(self):
        documents = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                filepath = os.path.join(root, name)
                with open(filepath) as f:
                    link = f.readline().strip()
                    body = f.read()
                    title, text = html_to_text(link, body)
                    if text:
                        doc = self.create_document(link, title, link, text)
                        documents.append(doc)
        return documents


def main():
    
    parser = argparse.ArgumentParser(description='Load the given web site export into Weaviate')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to extracted web site export directory')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8080", help='Weaviate database URL')
    parser.add_argument('-c', '--class-prefix', type=str, default="Web", help='Class prefix in Weaviate. The full class name will be "<prefix>_Node".')
    parser.add_argument('-r', '--remove-existing', default=False, action=argparse.BooleanOptionalAction, help='Remove existing "<prefix>_Node" class in Weaviate before starting.')
    parser.add_argument('-d', '--debug', default=False, action=argparse.BooleanOptionalAction, help='Print debugging information, such as the message content.')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load the Slack archive from disk and process it into documents
    loader = ArchivedWebSiteLoader(args.input)
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # Index the documents in Weaviate
    indexer = Indexer(args.weaviate_url, args.class_prefix, args.remove_existing)
    indexer.index(documents)

if __name__ == '__main__':
    main()
