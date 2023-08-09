#
# This file contains a Scrapy spider. To run it:
#
#   scrapy runspider spider.py
#
import re
from pathlib import Path

from scrapy import Request, Item
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector

import bs4 as bs
import html2text

URL_PREFIX = "https://www.janelia.org"
OUTPUT_PATH = "./data/janelia.org"
DEBUG = False
IGNORED_CLASSES = ['panels-ipe-label']

h = html2text.HTML2Text()
h.ignore_links = True
h.images_to_alt = True
h.single_line_break = True

class MySpider(CrawlSpider):
    """ Spider that crawls janelial.org and saves content 
        to disk in Markdown format for consumption by an LLM.
    """
    name = "janelia.org"
    allowed_domains = ["www.janelia.org"]
    start_urls = [URL_PREFIX]
    rules = (
        Rule(LinkExtractor(allow=(r".*",)), callback="parse_item"),
    )
    seen = set()

    def get_path(self, url):
        path = url.replace(URL_PREFIX, '')
        path = path.split("#")[0]
        path = path.split("?")[0]

        if "/search" in path or "/node" in path:
            if DEBUG: print(f"Skipping: {url}")
            return None

        if path in self.seen:
            if DEBUG: print(f"Already seen: {url}")
            return None

        return path


    def parse_item(self, response):

        url = response.url
        path = self.get_path(url)
        self.seen.add(path)

        if not path: return

        headers = response.headers.to_unicode_dict()
        content_type = str(headers['Content-Type'])
        if not content_type.startswith('text/html'):
            print(f"Content type {content_type} is not HTML: {url}")
            return

        # Recurse into other links
        item = Item()
        sel = Selector(response)
        item_urls = sel.xpath(""".//*/a/@href""").getall()
        for item_url in item_urls:
            if item_url.startswith("/"):
                abs_url = self.start_urls[0] +""+ item_url
                if self.get_path(abs_url):
                    yield Request(abs_url, callback=self.parse_item)

        # Extract content
        body = response.selector.css(".content-section").get()
        if not body:
            print(f"No content found: {url}")
        else:
            soup = bs.BeautifulSoup(body,'lxml')
            # certain classes contain metadata that is hidden
            # on janelia.org, this eliminates them from the output text
            for css_class in IGNORED_CLASSES:
                for div in soup.find_all("div", {'class':css_class}):
                    div.decompose()
            text = h.handle(str(soup))

            # Save to file
            filename = OUTPUT_PATH + path
            with Path(filename) as p:
                p.mkdir(parents=True, exist_ok=True)
                c = p / "content"
                c.write_text(text)
                print(f"Saved {path}")

