import os
import json
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from sklearn.preprocessing import normalize

# Load job postings data
with open('jobPostings.json', 'r') as file:
    jobPostings = json.load(file)

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Initialize Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Set collection name
collectionName = "resume_job"

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file and clean it."""
    try:
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        text = text.strip().lower()
        print("Extracted Resume Text:\n", text[:500])  # Print first 500 chars for debugging
        return text
    except Exception as err:
        print(f"Error extracting text from PDF: {err}")
        return ""

def generate_embeddings(text):
    """Generate and normalize a 384-dimensional embedding vector."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return normalize([embedding])[0].tolist()

def store_embeddings_in_chromadb(job_postings):
    """Store job embeddings in ChromaDB."""
    job_embeddings = []
    metadatas = [{"source": "jobPosting", "jobDescription": job["jobDescription"]} for job in job_postings]
    job_texts = [job["jobTitle"] + " - " + job["jobDescription"] for job in job_postings]
    ids = [str(i) for i in range(len(job_postings))]

    for job in job_postings:
        embedding = generate_embeddings(job["jobDescription"].lower())
        job_embeddings.append(embedding)

    try:
        collection = client.get_or_create_collection(name=collectionName)
        collection.add(ids=ids, documents=job_texts, embeddings=job_embeddings, metadatas=metadatas)
        print("Stored job embeddings in Chroma DB.")
    except Exception as error:
        print(f"Error storing embeddings in Chroma DB: {error}")

def perform_similarity_search(collection, resume_embedding):
    """Perform similarity search for top 5 matching jobs."""
    results = collection.query(query_embeddings=[resume_embedding], n_results=5)
    return results

def main():
    print("Welcome to the Smart Job Posting Recommendation System!")
    store_embeddings_in_chromadb(jobPostings)

    file_path = input("Enter the path to your resume PDF: ")
    resume_text = extract_text_from_pdf(file_path)

    if resume_text:
        resume_embedding = generate_embeddings(resume_text)
        collection = client.get_or_create_collection(name=collectionName)  # Ensure correct retrieval
        results = perform_similarity_search(collection, resume_embedding)

        if results and results["ids"]:
            print("Recommended Job Postings:")
            for index, id in enumerate(results["ids"][0]):
                job_data = jobPostings[int(id)]
                print(f"Top {index + 1}: {job_data['jobTitle']} at {job_data['company']}")
        else:
            print("No suitable jobs found.")
    else:
        print("No text retrieved from the resume.")

if __name__ == "__main__":
    main()
