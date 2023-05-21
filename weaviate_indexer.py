import sys
import logging
import warnings

from langchain.embeddings import OpenAIEmbeddings
from llama_index import PromptHelper, ServiceContext, LangchainEmbedding, GPTVectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext

import weaviate

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_INPUT_SIZE = 4096
NUM_OUTPUT = 256
MAX_CHUNK_OVERLAP = 20


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
            class_name = f"{class_prefix}_Node"
            logger.warning(f"Deleting {class_name} class in Weaviate")
            client.schema.delete_class(class_name)

        # Create LLM embedding model
        embed_model = LangchainEmbedding(OpenAIEmbeddings())
        prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, MAX_CHUNK_OVERLAP)
        service_context = ServiceContext.from_defaults(embed_model=embed_model, prompt_helper=prompt_helper)

        # Embed the documents and persist the embeddings into Weaviate    
        logger.info("Creating GPT vector store index")
        vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=class_prefix)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

        logger.info(f"Completed indexing into '{class_prefix}_Node'")
