import os
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def retrieve_similar_docs(question, top_k=3, docs_dir="docs"):
    model = get_embedding_model()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(current_dir, docs_dir)
    
    documents = []
    filenames = []
    
    if not os.path.exists(docs_path):
        return "No knowledge base found."
    
    for filename in os.listdir(docs_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(docs_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                    filenames.append(filename)
    
    if not documents:
        return "No documents found in knowledge base."
    
    # Generate embeddings for the question
    question_embedding = model.encode(question)
    
    # Generate embeddings for all documents
    doc_embeddings = model.encode(documents)
    
    # Calculate cosine similarity between question and each document
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = cosine_similarity(question_embedding, doc_embedding)
        similarities.append((similarity, i, filenames[i]))
    
    # Sort by similarity (descending)
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Get top_k most similar documents
    top_docs = similarities[:top_k]
    
    # Concatenate the content of top documents
    context = ""
    for similarity_score, doc_idx, filename in top_docs:
        context += f"[Source: {filename} | Similarity: {similarity_score:.3f}]\n"
        context += documents[doc_idx]
        context += "\n\n---\n\n"
    
    return context.strip()


if __name__ == "__main__":
    question = input("Ask a question: ")
    context = retrieve_similar_docs(question, top_k=2)
    print(context)