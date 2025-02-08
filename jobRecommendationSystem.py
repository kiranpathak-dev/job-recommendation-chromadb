import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np

# Load dataset from a JSON file
with open('jobPostings.json', 'r') as file:
    jobPostings = json.load(file)

# Set collection name
collectionName = "job_collection"

# Initialize Chroma client (persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")

# Initialize Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(texts):
    """Generate and normalize embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract CLS token representation and normalize
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return normalized_embeddings.tolist()

def perform_similarity_search(collection, query_text):
    """Perform similarity search using embeddings."""
    query_embedding = generate_embeddings([query_text])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    if not results or len(results['ids'][0]) == 0:
        print(f"No job posts found similar to '{query_text}'")
        return []

    top_job_postings = []
    for idx, id_str in enumerate(results['ids'][0]):
        job = next((item for item in jobPostings if str(item['jobId']) == id_str), None)
        if job:
            top_job_postings.append({
                'id': id_str,
                'score': results['distances'][0][idx],
                'jobTitle': job['jobTitle'],
                'jobDescription': job['jobDescription'],
                'jobType': job.get('jobType', 'Unknown'),
                'company': job.get('company', 'Unknown')
            })

    return sorted(top_job_postings, key=lambda x: x['score'])

def main():
    query = "Frontend Developer"

    try:
        # 🔹 **Fix: Use correct way to check collections in ChromaDB v0.6.0**
        existing_collections = client.list_collections()
        if collectionName in existing_collections:
            client.delete_collection(name=collectionName)

        # Create a new collection
        collection = client.get_or_create_collection(name=collectionName)

        # Store job postings
        job_texts = [f"{job['jobTitle']}. {job['jobDescription']}. Job Type: {job.get('jobType', 'Unknown')}. Location: {job.get('location', 'Unknown')}" for job in jobPostings]
        ids = [str(job['jobId']) for job in jobPostings]

        # Generate and store embeddings
        embeddings_data = generate_embeddings(job_texts)
        collection.add(
            ids=ids,
            documents=job_texts,
            embeddings=embeddings_data
        )

        # Perform similarity search
        initial_results = perform_similarity_search(collection, query)

        # Log the top 3 job postings
        if initial_results:
            print("\n--- Top 3 Recommended Jobs ---\n")
            for index, item in enumerate(initial_results[:3]):
                print(f"🔹 **Job {index + 1}:** {item['jobTitle']} at {item['company']}")
                print(f"   📍 **Location:** {item['jobDescription']}")
                print(f"   📅 **Job Type:** {item['jobType']}")
                print(f"   📊 **Score:** {item['score']:.4f}")
                print("\n" + "-" * 40 + "\n")
        else:
            print("No jobs found based on the given query.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
