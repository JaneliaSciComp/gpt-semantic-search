import streamlit as st

st.set_page_config(page_title="JaneliaGPT - About", page_icon="ℹ️")

from state import init_state
init_state()

num_results = st.session_state["num_results"]

st.markdown("# About JaneliaGPT")
st.markdown(f"""

## How does JaneliaGPT work?
Content from the Janelia-Software Slack, the Janelia Wiki, and Janelia.org are translated into semantic vectors using OpenAI's embedding API 
and stored in a vector database. Your query is embedded as well and used to search the database for content that is 
semantically related to your query. The GPT language model tries to answer your question using the top {num_results} results 
(configurable under Settings).

## Why is the answer unrelated to my question?
At the moment this is just a proof of concept. It works brilliantly for some questions and fails spectacularly for others.
Please use the survey buttons to record your experience so that we can use the results to improve the search results in future iterations. 

## Why can't I find something that was posted recently?
Source data was downloaded on April 1, 2024. If the search proves useful, we can implement automated downloading and incremental indexing. 

## Where is the source code?
[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/JaneliaSciComp/gpt-semantic-search)
""")
