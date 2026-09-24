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
            "You are a strict evaluator. Analyze whether the answer is fully grounded "
            "in the context. Return a score from 0.0 to 1.0 where:\n"
            "- 1.0: Answer is completely supported by context with no hallucinations\n"
            "- 0.5: Answer is partially supported but contains some unsupported claims\n"
            "- 0.0: Answer contains significant hallucinations or is not supported by context\n\n"
            "Respond with ONLY the numeric score (e.g., '0.75').",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)
_ANSWER_RELEVANCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Analyze whether the answer directly addresses "
            "the question. Return a score from 0.0 to 1.0 where:\n"
            "- 1.0: Answer directly and completely addresses the question\n"
            "- 0.5: Answer partially addresses the question but is incomplete or tangential\n"
            "- 0.0: Answer does not address the question or is completely irrelevant\n\n"
            "Respond with ONLY the numeric score (e.g., '0.75').",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)
_CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Analyze whether the retrieved context is "
            "relevant and precise for answering the question. "
            "Return a score from 0.0 to 1.0 where:\n"
            "- 1.0: Context is highly relevant and precisely matches what's needed\n"
            "- 0.5: Context is somewhat relevant but contains noise or irrelevant information\n"
            "- 0.0: Context is irrelevant or does not help answer the question\n\n"
            "Respond with ONLY the numeric score (e.g., '0.75').",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
_CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Analyze whether the retrieved context contains "
            "all information needed to answer the question. "
            "Return a score from 0.0 to 1.0 where:\n"
            "- 1.0: Context contains all necessary information to answer the question completely\n"
            "- 0.5: Context contains some relevant information but is missing key details\n"
            "- 0.0: Context lacks the information needed to answer the question\n\n"
            "Respond with ONLY the numeric score (e.g., '0.75').",
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


def _score_from_response(prompt: ChatPromptTemplate, **inputs: str) -> float:
    """Extract a numeric score (0.0-1.0) from LLM response with robust parsing."""
    chain = prompt | _get_llm() | StrOutputParser()
    raw = chain.invoke(inputs).strip().lower()

    # Try to extract a number from the response
    import re
    number_match = re.search(r'(\d+\.?\d*)', raw)
    if number_match:
        try:
            score = float(number_match.group(1))
            # Clamp to 0.0-1.0 range
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Fallback: check for yes/no keywords
    if raw.startswith("yes"):
        return 1.0
    elif raw.startswith("no"):
        return 0.0

    # Default fallback
    return 0.5


def faithfulness(context: str, answer: str) -> float:
    """Score from 0.0 to 1.0: how well the answer is grounded in the context."""
    return _score_from_response(_FAITHFULNESS_PROMPT, context=context, answer=answer)


def answer_relevancy(question: str, answer: str) -> float:
    """Score from 0.0 to 1.0: how well the answer addresses the question."""
    return _score_from_response(_ANSWER_RELEVANCY_PROMPT, question=question, answer=answer)


def context_precision(question: str, context: str) -> float:
    """Score from 0.0 to 1.0: how relevant and precise the context is for the question."""
    return _score_from_response(_CONTEXT_PRECISION_PROMPT, question=question, context=context)


def context_recall(question: str, context: str) -> float:
    """Score from 0.0 to 1.0: how well the context covers the information needed."""
    return _score_from_response(_CONTEXT_RECALL_PROMPT, question=question, context=context)


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
