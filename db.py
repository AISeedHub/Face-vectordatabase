import chromadb

# Initialize ChromaDB client
# For persistent storage, specify a path:
client = chromadb.PersistentClient(path="./chroma_db")
# For in-memory (cleared on script exit):
# client = chromadb.Client()


# Create a collection or get it if it already exists
# The collection name is "face_embeddings"
# The metadata field "hnsw:space" specifies the distance metric (L2 for Euclidean, cosine)
# For face embeddings, 'cosine' similarity is often preferred.
COLLECTION_NAME = "face_embeddings"
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded.")
except:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # L2 is default, cosine is good for embeddings
    )
    print(f"Collection '{COLLECTION_NAME}' created.")