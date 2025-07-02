import streamlit as st

PERSIST_KEYS = ["model_options","model","search_alpha","num_results","temperature","class_prefix","query","survey_complete","response","response_error","db_id","last_processed_query","enable_query_caching","show_progress","debug_mode","enable_unit_testing","verbose_test_output","auto_run_tests","test_framework"]
DEFAULT_CLASS_PREFIX = "Janelia"

@st.cache_resource
def get_models():
    """ Returns a list of available GPT models.
    """
    from openai import OpenAI
    
    client = OpenAI()
    model_res = client.models.list()
    models = [model.id for model in model_res.data]
    return sorted(models)


def init_state():
    """ Initialize the session state if it has not already been initialized.
    """
    # this is the magic that copies certain widget states into the 
    # user session state for persistance across page changes
    st.session_state.update({
        key: value
        for key, value in st.session_state.items()
        if key in PERSIST_KEYS
    })
    # initial state values
    if "model" not in st.session_state:
        st.session_state.update({
            "model_options": get_models(),
            "model": "gpt-4o-mini",
            # "model": "gpt-3.5-turbo",
            "search_alpha": 55,
            "num_results": 3,
            "temperature": 0,
            "class_prefix": DEFAULT_CLASS_PREFIX,
            "response": None,
            

            # Search-specific state
            "survey_complete": True,
            "query": "",
            "response_error": False,
            "db_id": None,
            "last_processed_query": "",

            "admin_toggle": True,
            
            # Performance settings (kept minimal)
            "enable_query_caching": True,
            "show_progress": True,
            "debug_mode": False,
            
            # Testing settings
            "enable_unit_testing": True,
            "verbose_test_output": False,
            "auto_run_tests": False,
            "test_framework": "deepeval"
            
        })
        #print("initialized session state: ",st.session_state)
