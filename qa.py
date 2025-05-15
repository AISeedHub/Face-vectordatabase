from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from db import collection
import json

llm = OllamaLLM(model="llama3", temperature=0.1)

# Helper to get collection attributes
def get_collection_attributes():
    sample = collection.get(limit=1, include=["metadatas"])
    if sample and "metadatas" in sample and sample["metadatas"]:
        return list(sample["metadatas"][0].keys())
    return []

def search_faces(query):
    """
    Search the face database based on a natural language query.
    """
    try:
        # Retrieve collection attributes
        attributes = get_collection_attributes()
        print(f"Collection attributes: {attributes}")

        # Add attributes to the prompt as context
        prompt_template = f"""
You are a system that interprets natural language queries to filter a face database.
The database contains metadata about faces, such as {', '.join(attributes)}.
Only return the filtering criteria in JSON format, without any additional text or explanation.

Given the query: "{{query}}"
Return the filtering criteria in JSON format. For example:
Query: "show men face"
Output: {{ "gender": "male" }}
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        qa_chain = prompt | llm

        # Use the LLM to interpret the query
        filtering_criteria = qa_chain.invoke({"query": query})
        print(f"Filtering criteria: {filtering_criteria}")

        criteria = json.loads(filtering_criteria)
        print(f"Parsed criteria: {criteria}")

        # Patch: If more than one key, wrap in $and
        if len(criteria) > 1:
            criteria = {"$and": [{k: v} for k, v in criteria.items()]}

        # Query the database using the criteria
        results = collection.get(
            where=criteria,
            include=["metadatas"]
        )

        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    query = "show man face and name contain 'Jin'"
    results = search_faces(query)
    if results:
        print(f"Found {len(results['metadatas'])} matching faces:")
        for result in results['metadatas']:
            print(result)
    else:
        print("No matching faces found.")