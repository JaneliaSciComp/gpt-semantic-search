import sys
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List

from ollama_client import SimpleOllamaAPI

import weaviate

warnings.simplefilter("ignore", ResourceWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBED_MODEL_NAME="bge-m3:567m"
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
        self.ollama = SimpleOllamaAPI()

    def index(self, documents):
        """Index all documents at once (legacy method)."""
        return self.index_batch(documents, batch_number=1, total_batches=1)

    def index_batch(self, documents, batch_number=1, total_batches=1, progress_callback=None):
        """Index a batch of documents with progress tracking."""
        if not documents:
            logger.warning("No documents to index in batch")
            return

        # Connect to Weaviate database
        client = weaviate.Client(self.weaviate_url)

        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            raise ConnectionError(f"Weaviate is not live at {self.weaviate_url}")

        if not client.is_ready():
            logger.error(f"Weaviate is not ready at {self.weaviate_url}")
            raise ConnectionError(f"Weaviate is not ready at {self.weaviate_url}")

        logger.info(f"Connected to Weaviate at {self.weaviate_url} (Version {client.get_meta()['version']})")

        # Delete existing data in Weaviate (only for first batch)
        class_prefix = self.class_prefix
        if self.delete_database and batch_number == 1:
            class_name = _class_name(class_prefix)
            logger.warning(f"INDEXING: Deleting {class_name} class in Weaviate")
            client.schema.delete_class(class_name)

            logger.info(f"INDEXING: Creating {class_name} class in Weaviate")
            create_schema(client, class_prefix)
        elif batch_number == 1:
            # Ensure schema exists even if not deleting
            create_schema(client, class_prefix)

        # Direct indexing without LlamaIndex    
        logger.info(f"INDEXING: Processing batch {batch_number}/{total_batches} with {len(documents)} documents")
        
        # Process documents directly
        class_name = _class_name(class_prefix)
        for i, doc in enumerate(documents):
            try:
                # Get embedding for document text
                logger.debug(f"Computing embedding for document {i+1}/{len(documents)}: {doc.extra_info.get('title', 'Untitled')}")
                embedding = self.ollama.get_embedding(doc.text)
                
                if embedding is None or len(embedding) == 0:
                    logger.error(f"Failed to compute embedding for document {doc.doc_id}")
                    continue
                    
                logger.debug(f"Successfully computed embedding with {len(embedding)} dimensions")
                
                # Prepare document data
                doc_data = {
                    "text": doc.text,
                    "ref_doc_id": doc.doc_id,
                    "title": doc.extra_info.get("title", ""),
                    "link": doc.extra_info.get("link", ""),
                    "source": doc.extra_info.get("source", ""),
                    "scraped_at": doc.extra_info.get("scraped_at", 0),
                    "_node_content": doc.to_json()
                }
                
                # Add to Weaviate with embedding
                client.data_object.create(
                    data_object=doc_data,
                    class_name=class_name,
                    vector=embedding
                )
                logger.debug(f"Successfully indexed document {doc.doc_id} with vector embedding")
                
            except Exception as e:
                logger.error(f"Error indexing document {doc.doc_id}: {str(e)}")
                continue

        logger.info(f"INDEXING: Completed batch {batch_number}/{total_batches} - {len(documents)} documents indexed into '{class_prefix}_Node'")
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback("indexing", batch_number, total_batches)

    def get_latest_timestamp(self, source: str = None) -> float:
        """Query Weaviate for the most recent scraped_at timestamp.
        
        Args:
            source: Optional source filter (e.g., "Slack"). If None, queries all sources.
        
        Returns:
            Unix timestamp of the most recent document, or 0.0 if no documents found
        """
        client = weaviate.Client(self.weaviate_url)
        
        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            return 0.0
        
        class_name = _class_name(self.class_prefix)
        
        try:
            query = client.query.aggregate(class_name)
            
            if source:
                query = query.with_where({
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": source
                })
            
            result = query.with_fields("scraped_at { maximum }").do()
            
            try:
                max_timestamp = result["data"]["Aggregate"][class_name][0]["scraped_at"]["maximum"]
                source_msg = f" for {source}" if source else ""
                logger.info(f"Found latest timestamp{source_msg} in database: {max_timestamp} ({datetime.fromtimestamp(max_timestamp)})")
                return float(max_timestamp)
            except (KeyError, IndexError):  
                source_msg = f" for {source}" if source else ""
                logger.info(f"No documents{source_msg} found in database or no scraped_at timestamps")
                return 0.0  
                
        except Exception as e:
            source_msg = f" for {source}" if source else ""
            logger.error(f"Error querying latest timestamp{source_msg}: {e}")
            return 0.0

    def get_latest_slack_timestamp(self) -> float:
        """Query Weaviate for the most recent scraped_at timestamp for Slack documents.
        
        Returns:
            Unix timestamp of the most recent Slack document scraping, or 0.0 if no documents found
        """
        return self.get_latest_timestamp(source="Slack")
    
    def get_all_filesystem_documents(self) -> List[Dict[str, Any]]:
        """Query Weaviate for all filesystem documents with their metadata.
        
        Returns:
            List of document dictionaries with file_path, file_mtime, file_hash, etc.
        """
        client = weaviate.Client(self.weaviate_url)
        
        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            return []
        
        class_name = _class_name(self.class_prefix)
        
        try:
            # Query for all documents with source="Filesystem"
            result = client.query.get(class_name, [
                "file_path", "title", "file_mtime", "file_size", "file_hash", 
                "indexed_at", "processing_method"
            ]).with_where({
                "path": ["source"],
                "operator": "Equal", 
                "valueText": "Filesystem"
            }).with_limit(10000).do()  # Large limit to get all documents
            
            documents = []
            if "data" in result and "Get" in result["data"]:
                for doc in result["data"]["Get"][class_name]:
                    documents.append(doc)
            
            logger.info(f"Found {len(documents)} filesystem documents in Weaviate")
            return documents
            
        except Exception as e:
            logger.error(f"Error querying filesystem documents: {e}")
            return []
    
    def delete_document_by_path(self, file_path: str) -> bool:
        """Delete a document from Weaviate by its file path (doc_id).
        
        Args:
            file_path: The file path used as doc_id
            
        Returns:
            True if deletion was successful, False otherwise
        """
        client = weaviate.Client(self.weaviate_url)
        
        if not client.is_live():
            logger.error(f"Weaviate is not live at {self.weaviate_url}")
            return False
        
        class_name = _class_name(self.class_prefix)
        
        try:
            # Query to find the document by file_path
            result = client.query.get(class_name, ["file_path"]).with_where({
                "path": ["file_path"],
                "operator": "Equal",
                "valueText": str(file_path)
            }).with_additional(["id"]).do()
            
            if "data" in result and "Get" in result["data"] and result["data"]["Get"][class_name]:
                # Delete all matching documents (should be just one)
                deleted_count = 0
                for doc in result["data"]["Get"][class_name]:
                    doc_id = doc["_additional"]["id"]
                    client.data_object.delete(
                        uuid=doc_id,
                        class_name=class_name
                    )
                    deleted_count += 1
                
                logger.info(f"Deleted {deleted_count} document(s) for file path: {file_path}")
                return True
            else:
                logger.debug(f"No document found for file path: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document for {file_path}: {e}")
            return False
    
    def delete_documents_by_paths(self, file_paths: List[str]) -> int:
        """Delete multiple documents from Weaviate by their file paths.
        
        Args:
            file_paths: List of file paths to delete
            
        Returns:
            Number of documents successfully deleted
        """
        deleted_count = 0
        for file_path in file_paths:
            if self.delete_document_by_path(file_path):
                deleted_count += 1
        
        logger.info(f"Batch deletion complete: {deleted_count}/{len(file_paths)} documents deleted")
        return deleted_count
