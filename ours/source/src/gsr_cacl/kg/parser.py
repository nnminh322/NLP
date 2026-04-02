"""Markdown table parsing utilities."""

from __future__ import annotations

import re


def parse_markdown_rows(table_md: str) -> list[list[str]]:
    """
    Parse a markdown table into rows of cell strings.

    Example:
        | A | B | C |
        |---|---|---|
        | 1 | 2 | 3 |

    Returns:
        list of rows (each row = list of cell strings), separator rows excluded.
    """
    lines = table_md.strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip markdown separator lines like |---|---|
        if re.match(r"^\|[\s\-:]+\|$", line) or re.match(r"^[\s\-:\|]+$", line):
            continue
        # Split by '|', strip whitespace
        cells = [cell.strip() for cell in line.split("|")]
        # Remove leading/trailing empty cells (from leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    return rows


def parse_number(cell: str) -> float | None:
    """
    Parse a numeric value from a cell string.
    Handles: "1,234", "$1,234", "(123)", "-123", "12.3%", "1.2B", "500M"
    Returns None if not numeric.
    """
    if not cell or not cell.strip():
        return None
    s = cell.strip()

    # Negative in parentheses: (123) → -123
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # Strip currency symbols and commas
    s = re.sub(r"[\$€£¥,]", "", s)

    # Strip percentage → multiply by 0.01
    pct = False
    if s.endswith("%"):
        pct = True
        s = s[:-1]

    # Scale suffixes: B, M, K
    scale = 1.0
    if s.endswith("B") or s.endswith("b"):
        scale = 1e9
        s = s[:-1]
    elif s.endswith("M") or s.endswith("m"):
        scale = 1e6
        s = s[:-1]
    elif s.endswith("K") or s.endswith("k"):
        scale = 1e3
        s = s[:-1]

    try:
        val = float(s)
        if neg:
            val = -val
        val *= scale
        if pct:
            val *= 0.01
        return val
    except ValueError:
        return None
