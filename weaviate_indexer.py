import sys
import logging
import warnings
from typing import Any, Dict, List

from langchain.embeddings import OpenAIEmbeddings
from llama_index import PromptHelper, ServiceContext, LangchainEmbedding, GPTVectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext

import weaviate

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1

# Copied from weaviate_indexer to: 
# 1) upgrade string->text for proper tokenization
# 2) set tokenization which defaults to whitespace for some reason
# 3) disable indexes on metadata json
NODE_SCHEMA: List[Dict] = [
    {
        "name": "ref_doc_id",
        "dataType": ["text"],
        "description": "The ref_doc_id of the Node"
    },
    {
        "name": "_node_content",
        "dataType": ["text"],
        "description": "Node content (in serialized JSON)",
        "indexFilterable": False,
        "indexSearchable": False,
        "tokenization": 'word'
    },
    {
        "name": "text",
        "dataType": ["text"],
        "description": "Full text of the node",
        "tokenization": 'word'
    },
    {
        "name": "title",
        "dataType": ["text"],
        "description": "The title of the document",
        "tokenization": 'word'
    },
    {
        "name": "link",
        "dataType": ["text"],
        "description": "HTTP link to the source document",
        "tokenization": 'field'
    },
    {
        "name": "source",
        "dataType": ["text"],
        "description": "Data source for the source document",
        "tokenization": 'field'
    }
]

def create_schema(client: Any, class_prefix: str) -> None:
    """Create schema."""
    # first check if schema exists
    schema = client.schema.get()
    classes = schema["classes"]
    existing_class_names = {c["class"] for c in classes}
    # if schema already exists, don't create
    class_name = _class_name(class_prefix)
    if class_name in existing_class_names:
        return

    properties = NODE_SCHEMA
    class_obj = {
        "class": _class_name(class_prefix),  # <= note the capital "A".
        "description": f"Class for {class_name}",
        "properties": properties,
    }
    client.schema.create_class(class_obj)


def _class_name(class_prefix: str) -> str:
    """Return class name."""
    return f"{class_prefix}_Node"

class Indexer():

    def __init__(self, weaviate_url, class_prefix, delete_database):
        self.weaviate_url = weaviate_url
        self.class_prefix = class_prefix
        self.delete_database = delete_database

    def index(self, documents):

        # Connect to Weaviate database
        client = weaviate.Client(self.weaviate_url)

        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            sys.exit(1)

        if not client.is_live():
            logger.error(f"Weaviate is not ready at {self.weaviate_url}")
            sys.exit(1)

        logger.info(f"Connected to Weaviate at {self.weaviate_url} (Version {client.get_meta()['version']})")

        # Delete existing data in Weaviate
        class_prefix = self.class_prefix
        if self.delete_database:
            class_name = _class_name(class_prefix)
            logger.warning(f"Deleting {class_name} class in Weaviate")
            client.schema.delete_class(class_name)

            logger.info(f"Creating {class_name} class in Weaviate")
            from llama_index.vector_stores.weaviate_utils import NODE_SCHEMA
            create_schema(client, class_prefix)

        # Create LLM embedding model
        embed_model = LangchainEmbedding(OpenAIEmbeddings())
        prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)
        service_context = ServiceContext.from_defaults(embed_model=embed_model, prompt_helper=prompt_helper)

        # Embed the documents and persist the embeddings into Weaviate    
        logger.info("Creating GPT vector store index")
        vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

        logger.info(f"Completed indexing into '{class_prefix}_Node'")
