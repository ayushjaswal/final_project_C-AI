from llm_controller import llm_call, llm_call_with_rag
from datetime import datetime

def answer_with_llm(prompt):
    prompt = f"Answer the following question: \n{prompt}"
    return llm_call(prompt)

def answer_with_rag(question):
    return llm_call_with_rag(question)

if __name__ == "__main__":
    question = input("Ask a question: ")
    
    answer = answer_with_llm(question)
    answer_rag = answer_with_rag(question)
    

    with open(f"outputs/LOG_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "a") as f:
        f.write("WITH LLM")
        f.write(f"Question: {question}\nAnswer: {answer}\n\n")
        f.write("WITH RAG")
        f.write(f"Question: {question}\nAnswer: {answer_rag}\n\n")
    
    print(f"ANSWER WITH LLM: {answer}")
    print(f"ANSWER WITH RAG: {answer_rag}")