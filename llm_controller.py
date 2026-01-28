import os
from openai import OpenAI
from dotenv import load_dotenv
from helpers import retrieve_similar_docs

# Load environment variables from .env file
load_dotenv()

# Lazy initialization of OpenAI client
_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client

def llm_call(prompt):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def llm_call_with_rag(question):
    context = retrieve_similar_docs(question)
    prompt = f"""
        Use ONLY the following context to answer the question.
        If the context is none, say "I don't know".

        Context:
        {context}

        Question:
        {question}
    """ 
    return llm_call(prompt)