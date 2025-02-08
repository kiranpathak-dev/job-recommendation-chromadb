# Smart Job Recommendation System

## Overview
The **Smart Job Recommendation System** is an AI-powered tool that leverages **natural language processing (NLP)** and **vector search** to recommend job postings based on a user's resume. It utilizes **ChromaDB**, **Hugging Face Transformers**, and **PyPDF2** to process job descriptions and resumes, generating **semantic embeddings** to match candidates with relevant job opportunities.

## Features
- Extracts text from PDF resumes using **PyPDF2**
- Generates embeddings using **sentence-transformers/all-MiniLM-L6-v2**
- Stores job posting embeddings in **ChromaDB**
- Performs **semantic search** to find the most relevant job postings
- Matches resumes with job descriptions using **cosine similarity**

## Project Structure
```
|-- jobRecommendationSystem.py     # Main script for job recommendations
|-- resumeMatchingSystem.py        # Resume matching system using ChromaDB
|-- jobPostings.json               # JSON file containing job postings
|-- testResume.pdf                 # Sample resume for testing
|-- README.md                      # Project documentation
```

## Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/job-recommendation-system.git
cd job-recommendation-system
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Job Recommendation System
```bash
python jobRecommendationSystem.py
```

### 5️⃣ Run the Resume Matching System
```bash
python resumeMatchingSystem.py
```

## Usage
### Running Job Recommendations
1. The system loads job postings from `jobPostings.json`.
2. It processes and embeds job descriptions using a Transformer model.
3. The system performs a **semantic similarity search** based on a user query (e.g., "Frontend Developer").
4. It displays the **top recommended jobs**.

### Running Resume Matching
1. The system **extracts text** from a PDF resume.
2. It generates a **resume embedding**.
3. It compares the resume embedding with stored **job embeddings** in ChromaDB.
4. It outputs **top-matching job postings**.

## Dependencies
- `chromadb`
- `torch`
- `transformers`
- `numpy`
- `scikit-learn`
- `PyPDF2`
