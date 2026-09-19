"""Prompt templates used by the Self-RAG graph nodes."""

from langchain_core.prompts import ChatPromptTemplate

GRADE_DOCUMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document "
            "to a user question.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the "
            "document is relevant to the question. It does not need to be a "
            "strict, exhaustive match - the goal is to filter out documents "
            "that are clearly unrelated.",
        ),
        (
            "human",
            "Retrieved document:\n\n{document}\n\nUser question: {question}",
        ),
    ]
)

GENERATE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the "
            "retrieved context below to answer the user's question. If the "
            "context does not contain the answer, say that you don't know. "
            "Keep the answer concise (max three sentences) and grounded "
            "strictly in the provided context.",
        ),
        (
            "human",
            "Context:\n\n{context}\n\nQuestion: {question}",
        ),
    ]
)

TRANSFORM_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions to improve retrieval from a vector "
            "database. Look at the input question and try to reason about "
            "the underlying semantic intent, then produce a clearer, more "
            "specific version of the question. Return only the rewritten "
            "question, with no additional commentary.",
        ),
        (
            "human",
            "Original question: {question}",
        ),
    ]
)
