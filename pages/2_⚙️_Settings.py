import streamlit as st

st.set_page_config(page_title="JaneliaGPT - Settings", page_icon="⚙️")

from state import init_state, persist
init_state()

PARAM_EXPLANATION = """
* The **Search Alpha** controls how the search balances BM25 (keyword) vs Dense (vector) search algorithms. At 0, only keywords 
are used and at 100 only vectors are used. In between, both are used and results are re-ranked with Reciprocal Rank Fusion.
* **Num Results** controls how many results are returned from the search and fed into the GPT model. Increasing this will increase 
the latency of the search.
* The GPT model's **Temperature** controls its creativity. It's generally best to keep this low to avoid hallucination. 
"""

st.markdown("# JaneliaGPT Settings")

st.selectbox("Model", st.session_state["model_options"], key=persist("model"))

col1, col2 = st.columns([1,1])
with col1:
    st.slider("Search Alpha", 0, 100, key=persist("search_alpha"))
    st.slider("Num Results", 0, 10, key=persist("num_results"))
    st.slider("Temperature", 0, 100, key=persist("temperature"))
with col2:
    st.markdown(PARAM_EXPLANATION)
    
st.text_input("Weaviate Class Prefix", key=persist("class_prefix"))

# st.checkbox("Checkbox", key=persist("checkbox"))
# st.radio("Radio", st.session_state["options"], key=persist("radio"))
# st.multiselect("Multiselect", st.session_state["options"], key=persist("multiselect"))
