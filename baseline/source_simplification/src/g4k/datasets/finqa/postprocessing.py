"""Postprocessing for FinQA of LLM answers."""

import logging

from g4k.datasets.finqa.finqa_parser import FinQAParser

logger = logging.getLogger(__name__)


def extract_answer(response_text: str | None, table: str) -> str:
    """Extract the answer from the response text."""
    if response_text is None:
        return "None"
    return extract_answer_finqa(response_text, table)


def extract_answer_finqa(response_text: str, table: str) -> str:
    """Extract the answer from the response text."""
    try:
        parser = FinQAParser(table)
        program_answer = parser.parse(response_text)
    except (TypeError, ValueError, ZeroDivisionError) as e:
        print(f"Error parsing response: {e}")
        program_answer = "None"
    return program_answer


def extract_answer_python(response_text: str, table: str) -> str:
    """Extract the answer from a Python expression."""
    row_ops = {
        "row_max": max,
        "row_min": min,
        "row_sum": sum,
        "row_avg": (lambda x: sum(x) / len(x)),
    }

    try:
        parser = FinQAParser(table)
        program_answer = eval(
            response_text,
            {k: lambda row_name, v=v: v(parser.read_table(row_name)) for k, v in row_ops.items()},
        )
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError, NameError):
        return "None"

    if isinstance(program_answer, bool):
        return "yes" if program_answer else "no"

    try:
        program_answer = float(program_answer)
    except (TypeError, ValueError):
        return "None"
    return str(program_answer)
