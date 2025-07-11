#!/usr/bin/env python
"""
Download spaces on the Janelia wiki and save to files on disk
Based on documentation at https://atlassian-python-api.readthedocs.io/confluence.html#get-page-info

To use this script you must create a Personal Access Token and save it into your environment:
https://wikis.janelia.org/plugins/personalaccesstokens/usertokens.action
"""

import os
from atlassian import Confluence

confluence_url = "https://wikis.janelia.org"
spaces = ['SCSW','SCS','ScientificComputing']
outpath = "./data/wiki"

confluence_pat = os.environ.get('CONFLUENCE_TOKEN')
confluence = Confluence(url=confluence_url, token=confluence_pat)


def parse_page(page):
    page_id = int(page['id'])
    path = page['_links']['webui']
    title = page['title']
    body = page['body']['view']['value']
    labels = [l['name'] for l in page['metadata']['labels']['results']]
    ancestors = [p['title'] for p in page['ancestors']]
    createdBy = page['history']['createdBy']['displayName']
    authors = [createdBy]
    if 'lastUpdated' in page['history']:
        lastUpdatedBy = page['history']['lastUpdated']['by']['displayName']
        if lastUpdatedBy != createdBy:
            authors.append(lastUpdatedBy)

    return page_id,path,title,body,labels,ancestors,authors


def get_page(page_id):
    # This only retrieves the createdBy and lastUpdatedBy for history
    page = confluence.get_page_by_id(page_id, status=None, version=None,
            expand="body.view,metadata.labels,ancestors,history,history.lastUpdated")
    return parse_page(page)


def get_link(path):
    return "%s%s" % (confluence_url, path)


limit = 50
num_pages = 0

for space in spaces:

    start = 0
    pages = []
    while True:
        pages_iter = confluence.get_all_pages_from_space(
            space,
            start=start,
            limit=limit,
            expand="body.view,metadata.labels,ancestors,history.lastUpdated"
        )

        if len(pages_iter) == 0:
            break

        start += len(pages_iter)

        for page in pages_iter:
            page_id,path,title,body,labels,ancestors,authors = parse_page(page)

            # Skip archived pages
            if "ARCHIVE_SC" in ancestors: continue
            if "ARCHIVE_SCSW" in ancestors: continue

            filepath = f"{outpath}/{space}/{page_id}"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, mode="wt") as f:
                f.write(get_link(path)+"\n")
                f.write(" / ".join(ancestors)+"\n")
                f.write(title+"\n")
                f.write(", ".join(authors)+"\n")
                f.write(", ".join(labels)+"\n")
                f.write(body)
                num_pages += 1

        # no more to fetch
        if len(pages_iter) < limit:
            break

    print('Wrote %d pages in %s to %s' % (num_pages,space,outpath))

