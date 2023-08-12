#
# Copied cross-page persistance support from https://gist.github.com/okld/0aba4869ba6fdc8d49132e6974e2e662
#
# Tried using https://gist.github.com/okld/8ca97ba892a7c20298fd099b196a8b4d as well, but ran into a lot of issues.
#
import streamlit as st

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in st.session_state:
        st.session_state[_PERSIST_STATE_KEY] = set()

    st.session_state[_PERSIST_STATE_KEY].add(key)

    return key


@st.cache_resource
def get_models():
    """ Returns a list of available GPT models.
    """
    import openai
    model_res = openai.Model.list()
    models = [model.id for model in model_res.data]
    return sorted(models)


def init_state():
    """ Initialize the session state if it has not already been initialized.
    """
    # this is the magic that copies certain widget states into the 
    # user session state for persistance across page changes
    if _PERSIST_STATE_KEY in st.session_state:
        st.session_state.update({
            key: value
            for key, value in st.session_state.items()
            if key in st.session_state[_PERSIST_STATE_KEY]
        })
    # initial state values
    if "model" not in st.session_state:
        st.session_state.update({
            "model_options": get_models(),
            "model": "gpt-4",
            "temperature": 0,
            "num_results": 3,
            "class_prefix": "Janelia",
            "search_alpha": 55
        })
