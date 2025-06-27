import sys
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptHelper, ServiceContext, GPTVectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext

import weaviate

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBED_MODEL_NAME="text-embedding-3-large"
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
    },
    {
        "name": "scraped_at",
        "dataType": ["number"],
        "description": "Unix timestamp when this document was scraped"
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
            create_schema(client, class_prefix)

        # Create LLM embedding model
        embed_model = OpenAIEmbedding(embed_batch_size=20, model=EMBED_MODEL_NAME)
        prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)
        service_context = ServiceContext.from_defaults(embed_model=embed_model, prompt_helper=prompt_helper)

        # Embed the documents and persist the embeddings into Weaviate    
        logger.info("Creating GPT vector store index")
        vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

        logger.info(f"Completed indexing into '{class_prefix}_Node'")

    def get_latest_timestamp(self) -> float:
        """Query Weaviate for the most recent scraped_at timestamp.
        
        Returns:
            Unix timestamp of the most recent document, or 0.0 if no documents found
        """
        client = weaviate.Client(self.weaviate_url)
        
        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            return 0.0
        
        class_name = _class_name(self.class_prefix)
        
        try:
            # Query for the maximum scraped_at timestamp
            result = (
                client.query
                .aggregate(class_name)
                .with_fields("scraped_at { maximum }")
                .do()
            )
            
            if (result.get("data") and 
                result["data"].get("Aggregate") and 
                result["data"]["Aggregate"].get(class_name) and
                len(result["data"]["Aggregate"][class_name]) > 0 and
                result["data"]["Aggregate"][class_name][0].get("scraped_at") and
                result["data"]["Aggregate"][class_name][0]["scraped_at"].get("maximum") is not None):
                
                max_timestamp = result["data"]["Aggregate"][class_name][0]["scraped_at"]["maximum"]
                logger.info(f"Found latest timestamp in database: {max_timestamp} ({datetime.fromtimestamp(max_timestamp)})")
                return float(max_timestamp)
            else:
                logger.info("No documents found in database or no scraped_at timestamps")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error querying latest timestamp: {e}")
            return 0.0

    def get_latest_slack_timestamp(self) -> float:
        """Query Weaviate for the most recent scraped_at timestamp for Slack documents.
        
        Returns:
            Unix timestamp of the most recent Slack document scraping, or 0.0 if no documents found
        """
        client = weaviate.Client(self.weaviate_url)
        
        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            return 0.0
        
        class_name = _class_name(self.class_prefix)
        
        try:
            # Query for Slack documents and get the maximum scraped_at value
            result = (
                client.query
                .aggregate(class_name)
                .with_where({
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": "Slack"
                })
                .with_fields("scraped_at { maximum }")
                .do()
            )
            
            if (result.get("data") and 
                result["data"].get("Aggregate") and 
                result["data"]["Aggregate"].get(class_name) and
                len(result["data"]["Aggregate"][class_name]) > 0 and
                result["data"]["Aggregate"][class_name][0].get("scraped_at") and
                result["data"]["Aggregate"][class_name][0]["scraped_at"].get("maximum") is not None):
                
                max_timestamp = result["data"]["Aggregate"][class_name][0]["scraped_at"]["maximum"]
                logger.info(f"Found latest Slack scraped_at timestamp: {max_timestamp} ({datetime.fromtimestamp(max_timestamp)})")
                return float(max_timestamp)
            else:
                logger.info("No Slack documents found in database")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error querying latest Slack scraped_at timestamp: {e}")
            return 0.0
