from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from db import collection  # Assuming you have a database collection for embeddings and metadata

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3")  # Replace "llama2" with the specific Ollama model you're using

# Define the prompt template for interpreting the query
prompt_template = """
You are a system that interprets natural language queries to filter a face database.
The database contains metadata about faces, such as gender, age, and other attributes.
Only return the filtering criteria in JSON format, without any additional text or explanation.

Given the query: "{query}"
Return the filtering criteria in JSON format. For example:
Query: "show men face"
Output: {{ "gender": "male" }}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create the LangChain pipeline
qa_chain = prompt | llm


def search_faces(query):
    """
    Search the face database based on a natural language query.
    """
    try:
        # Use the LLM to interpret the query
        filtering_criteria = qa_chain.invoke({"query": query})  # Pass the query to the LLM
        print(f"Filtering criteria: {filtering_criteria}")

        # Convert the filtering criteria to a dictionary
        import json
        criteria = json.loads(filtering_criteria)
        print(f"Parsed criteria: {criteria}")

        # Query the database using the criteria
        results = collection.get(
            where=criteria,  # Use the criteria to filter the database
            include=["metadatas"]  # Include metadata in the results
        )

        # Return the results
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    query = "show old man face, old man is over 60 years old"
    results = search_faces(query)
    if results:
        print(f"Found {len(results['metadatas'])} matching faces:")
        for result in results['metadatas']:
           print(result)
    else:
        print("No matching faces found.")