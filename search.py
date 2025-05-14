from get_embedding import get_face_embedding

def search_face(collection, query_image_path, top_n=3, threshold=0.7):
    print(f"\nSearching for faces similar to: {query_image_path}")
    query_embedding = get_face_embedding(query_image_path)

    if query_embedding is None:
        print("Could not generate embedding for the query image.")
        return

    if collection.count() == 0:
        print("The collection is empty. Cannot perform search.")
        return

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=['metadatas', 'distances']
    )

    if not results or not results.get('ids') or not results['ids'][0]:
        print("No similar faces found by query.")
        return
        
    print("\n--- Search Results ---")
    found_match = False
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        db_id = results['ids'][0][i]
        
        person_name = metadata.get('person_name', 'Unknown')
        source_image = metadata.get('source_image', 'Unknown')
        
        similarity_score = 1 - distance # For cosine distance

        print(f"  Candidate {i+1}: ID: {db_id}, Person: {person_name} (from {source_image})")
        print(f"    Cosine Distance: {distance:.4f} (Similarity: {similarity_score:.4f})")

        if similarity_score >= threshold:
            print(f"    MATCH FOUND! (Similarity {similarity_score:.4f} >= Threshold {threshold:.4f})")
            found_match = True
        else:
            print(f"    No match (Similarity {similarity_score:.4f} < Threshold {threshold:.4f})")
    
    if not found_match:
        print("No match found above the similarity threshold.")