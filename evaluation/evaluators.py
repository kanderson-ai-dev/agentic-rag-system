"""Evaluation evaluators.

Provides LangSmith-compatible evaluators (`llm_as_judge_evaluator`,
`groundedness_evaluator`, `relevance_evaluator`) and self-contained RAGAS-style
metric functions (`faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`), all implemented via LLM-as-judge so there is no external
`ragas` dependency.
"""

import json
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.graph.prompts import LLM_JUDGE_SYSTEM_PROMPT

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only 'yes' or 'no': is the answer fully supported by the "
            "context, with no unsupported claims?",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)
_ANSWER_RELEVANCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only 'yes' or 'no': does the answer directly address the question?",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)
_CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only 'yes' or 'no': is the retrieved context relevant and "
            "precise for answering the question?",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
_CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only 'yes' or 'no': does the retrieved context contain the "
            "information needed to answer the question?",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

LLM_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LLM_JUDGE_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)


@lru_cache
def _get_llm():
    from app.core.config import get_settings
    from app.services.llm import build_chat_model

    return build_chat_model(get_settings())


def _yes_no(prompt: ChatPromptTemplate, **inputs: str) -> float:
    chain = prompt | _get_llm() | StrOutputParser()
    raw = chain.invoke(inputs).strip().lower()
    return 1.0 if raw.startswith("yes") else 0.0


def faithfulness(context: str, answer: str) -> float:
    """1.0 if the answer is fully grounded in the context, else 0.0."""
    return _yes_no(_FAITHFULNESS_PROMPT, context=context, answer=answer)


def answer_relevancy(question: str, answer: str) -> float:
    """1.0 if the answer directly addresses the question, else 0.0."""
    return _yes_no(_ANSWER_RELEVANCY_PROMPT, question=question, answer=answer)


def context_precision(question: str, context: str) -> float:
    """1.0 if the retrieved context is precise for the question, else 0.0."""
    return _yes_no(_CONTEXT_PRECISION_PROMPT, question=question, context=context)


def context_recall(question: str, context: str) -> float:
    """1.0 if the retrieved context covers the answer, else 0.0."""
    return _yes_no(_CONTEXT_RECALL_PROMPT, question=question, context=context)


def _extract_qa(run: Any) -> tuple[str, str, str]:
    question = run.inputs["question"]
    outputs = run.outputs
    if isinstance(outputs, dict):
        answer = outputs.get("answer", "")
        context = "\n\n".join(outputs.get("contexts", []))
    else:
        answer = outputs
        context = ""
    return question, answer, context


def groundedness_evaluator(run: Any, example: Any) -> dict[str, Any]:
    """Groundedness (0/1) via LLM-as-judge against retrieved contexts."""
    _, answer, context = _extract_qa(run)
    return {"key": "groundedness", "score": faithfulness(context, answer)}


def relevance_evaluator(run: Any, example: Any) -> dict[str, Any]:
    """Answer relevancy (0/1) via LLM-as-judge."""
    question, answer, _ = _extract_qa(run)
    return {"key": "relevance", "score": answer_relevancy(question, answer)}


def llm_as_judge_evaluator(run: Any, example: Any) -> dict[str, Any]:
    """Rubric LLM-as-judge (1-5) for correctness, usefulness and safety."""
    question, answer, _ = _extract_qa(run)

    chain = LLM_JUDGE_PROMPT | _get_llm() | StrOutputParser()
    raw = chain.invoke({"question": question, "answer": answer})
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"correctness": 3, "usefulness": 3, "safety": 3, "justification": raw}

    average = (
        scores.get("correctness", 3) + scores.get("usefulness", 3) + scores.get("safety", 3)
    ) / 3
    return {
        "key": "llm_as_judge",
        "score": average,
        "comment": scores.get("justification", ""),
    }
