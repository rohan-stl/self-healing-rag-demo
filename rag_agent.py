import os
from typing import TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str
    grade: str
    retry_count: int


if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in your environment or .env file")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)


def retrieve(state: AgentState) -> AgentState:
    question = state.get("rewritten_question") or state["question"]
    print(f"\n[RETRIEVE] Searching for: {question}")

    try:
        results = collection.query(
            query_texts=[question],
            n_results=3,
        )
        documents = results.get("documents", [[]])[0]
        print(f"[RETRIEVE] Found {len(documents)} chunks.")
    except Exception as error:
        print(f"[RETRIEVE] ChromaDB query failed: {error}")
        documents = []

    return {**state, "documents": documents}



def generate(state: AgentState) -> AgentState:
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    print(f"\n[GENERATE] Generating answer for: {question}")

    if not documents:
        return {**state, "answer": "No relevant documents were found.", "grade": "FAIL"}

    context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)])

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on the provided documents. "
        "If the documents do not contain enough information to answer the question, say so honestly. "
        "Do not use any knowledge outside of the provided documents."
    )

    user_prompt = f"""Documents:
{context}

Question: {question}

Answer based only on the documents above:"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        answer = response.content.strip()
        print(f"[GENERATE] Answer generated ({len(answer)} chars).")
    except Exception as error:
        print(f"[GENERATE] LLM call failed: {error}")
        answer = "An error occurred while generating the answer."

    return {**state, "answer": answer}



def grade_answer(state: AgentState) -> AgentState:
    question = state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")
    print("\n[GRADE] Evaluating answer quality...")

    context = "\n\n".join(documents) if documents else "No documents."

    grading_prompt = f"""You are a strict grader evaluating whether an answer is grounded in the provided documents.

Documents:
{context}

Question:
{question}

Answer:
{answer}

Return only one word: PASS or FAIL.
- PASS if the answer is fully supported by the documents.
- FAIL if it is hallucinated, off-topic, unsupported, or incomplete.
"""

    try:
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        grade = response.content.strip().upper()
        if grade not in {"PASS", "FAIL"}:
            grade = "FAIL"
    except Exception as error:
        print(f"[GRADE] LLM grading failed: {error}")
        grade = "FAIL"

    print(f"[GRADE] Result: {grade}")
    return {**state, "grade": grade}



def rewrite_question(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0) + 1
    original_question = state["question"]
    current_answer = state.get("answer", "")

    print(f"\n[REWRITE] Rewriting question (attempt {retry_count})...")

    rewrite_prompt = f"""Rewrite this question to improve retrieval from a vector database.
Make it more specific and focused, but keep the same intent.

Original question: {original_question}
Current answer that failed grounding: {current_answer}

Return only the rewritten question."""

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten_question = response.content.strip()
    except Exception as error:
        print(f"[REWRITE] Rewrite failed: {error}")
        rewritten_question = original_question

    print(f"[REWRITE] New question: {rewritten_question}")
    return {**state, "rewritten_question": rewritten_question, "retry_count": retry_count}



def give_up(state: AgentState) -> AgentState:
    print("\n[GIVE_UP] Max retries reached. Returning fallback response.")
    return {
        **state,
        "answer": "I don't know based on the provided documents.",
        "grade": "FAIL",
    }



def route_after_grade(state: AgentState) -> str:
    if state.get("grade") == "PASS":
        return "end"
    if state.get("retry_count", 0) >= 2:
        return "give_up"
    return "rewrite_question"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_answer", grade_answer)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("give_up", give_up)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_answer")
workflow.add_conditional_edges(
    "grade_answer",
    route_after_grade,
    {
        "end": END,
        "rewrite_question": "rewrite_question",
        "give_up": "give_up",
    },
)
workflow.add_edge("rewrite_question", "retrieve")
workflow.add_edge("give_up", END)

app = workflow.compile()


if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    if not question:
        print("Please enter a question.")
    else:
        initial_state: AgentState = {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0,
        }
        result = app.invoke(initial_state)
        print("\nFinal Answer:\n")
        print(result.get("answer", "No answer generated."))
