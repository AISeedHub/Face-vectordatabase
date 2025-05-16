import chromadb
import weaviate

COLLECTION_NAME = "face_embeddings"

def init_chorma_db():
    """
    Initialize the database connection.
    This function is called at the start of the script to ensure the database is ready for use.
    """
    # Initialize ChromaDB client
    # For persistent storage, specify a path:
    client = chromadb.PersistentClient(path="./chroma_db")
    # Create a collection or get it if it already exists
    # The collection name is "face_embeddings"
    # The metadata field "hnsw:space" specifies the distance metric (L2 for Euclidean, cosine)
    # For face embeddings, 'cosine' similarity is often preferred.
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded.")

        return collection
    except:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # L2 is default, cosine is good for embeddings
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
        return collection
    
def init_weaviate_db():
    """
    Initialize the Weaviate database connection.
    This function is called at the start of the script to ensure the database is ready for use.
    """
    # Initialize Weaviate client
    client = weaviate.connect_to_local()
    
    try:
        # Check if the class exists
        collection = client.collections.get(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded.")
        return collection
    except weaviate.exceptions.WeaviateException:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        # Create a new class for the face embeddings
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
        
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
        return collection
    
def close_db(client):
    """
    Close the database connection.
    This function is called at the end of the script to ensure the database is properly closed.
    """
    client.close()
    print("Database connection closed.")