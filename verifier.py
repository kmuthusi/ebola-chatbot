"""Check that answer citations refer to retrieved PDF pages."""

import re

CITATION_RE = re.compile(r"\[([^\[\],]+\.pdf),\s*page\s+(\d+)\]", re.IGNORECASE)


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    if answer.strip() == "I cannot find this in the source documents.":
        return {"verified_answer": answer, "issues": []}
    cited = {(match.group(1), int(match.group(2))) for match in CITATION_RE.finditer(answer)}
    allowed = {(item.get("source"), item.get("page")) for item in evidence}
    issues = []
    if not cited:
        issues.append("No document and page citations found in answer.")
    invalid = cited - allowed
    if invalid:
        issues.append(f"Citations not in evidence: {sorted(invalid)}")
    return {"verified_answer": answer, "issues": issues}
