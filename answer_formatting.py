"""Make generated lists readable before citation and claim checks."""

import re


def format_answer_markdown(answer: str) -> str:
    """Put inline bullet and numbered items on separate Markdown lines."""
    answer = re.sub(
        r"[ \t]*\n\s*(\[[^\[\]\n]+\.pdf,\s*page\s+\d+\])",
        r" \1",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"[ \t]*•[ \t]*", "\n- ", answer)
    answer = re.sub(r"(?<=[\].:;])\s+(?=\d+[.)]\s+[A-Za-z])", "\n", answer)
    answer = re.sub(r"(?m)(?<!\n)\n(?=(?:[-*]|\d+[.)])\s)", "\n\n", answer)
    return answer.strip()
