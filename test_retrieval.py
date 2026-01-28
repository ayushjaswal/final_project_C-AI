"""
Test script to verify RAG implementation
"""
from helpers import retrieve_similar_docs

# Test the retrieval function
test_questions = [
    "What is FIPA ACL?",
    "What are FIPA performatives?",
    "What is MCP protocol?",
    "Explain RAG",
]

print("=" * 80) 
print("Testing Document Retrieval with Cosine Similarity")
print("=" * 80)

for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 80)
    context = retrieve_similar_docs(question, top_k=2)
    print(context)
    print("\n" + "=" * 80)
