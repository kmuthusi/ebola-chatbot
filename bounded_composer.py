"""Compose source-bound answers from retrieved PDF passages."""

import json
import os
import re

import ollama
from query_planning import classification_target, make_search_plan

DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def _claim_limit(question: str) -> int:
    """Reserve longer answers for broad or explicitly multi-part questions."""
    plan = make_search_plan(question)
    if plan.exposure:
        return 2
    if plan.comparison:
        return 2
    if re.search(r"\b(?:deadly|lethal|fatality|mortality|death rate)\b", question, re.I):
        return 1
    if plan.broad:
        return 5
    if re.match(r"\s*(?:how|which|can|could|is|are|does|do|what does|what should)\b", question, re.I) \
            and not re.search(r"\b(?:and|or)\b|\?.+\?", question, re.I):
        return 3
    return 4


def _make_prompt(question: str, evidence: list[dict], feedback: str = "") -> str:
    plan = make_search_plan(question)
    topic_request = not question.rstrip().endswith("?") and len(plan.subject.split()) <= 3
    if plan.exposure:
        scope = ("Write exactly two simple, positive claims. First, state only how Source 1 describes the named activity in its setting; do not add who performed it or assert that every instance transmits infection. "
                 "Second, state only the route of transmission described in Source 2. Do not mention the named activity or whether that route applies to it in the second claim. "
                 "Put no extra sentence in either claim. Do not discuss what the documents do not explicitly say. ")
    elif plan.comparison:
        left, right = plan.comparison
        scope = (f"Compare {left} and {right} directly. Begin with one short difference explicitly supported by Source 1. "
                 "Add one short shared feature only if useful and directly stated. A shared feature is not a difference. "
                 "Do not add host range, severity, outcomes, or transmission unless the question asks about them. "
                 "Do not claim different routes, effects, or outcomes when the sources describe the same one for both. "
                 "If no difference is supported, return an empty list. ")
    elif classification_target(question) and evidence and evidence[0].get("is_definition"):
        scope = ("State the category or type of the named entity from Source 1 in one short claim. "
                 "Use the subject of the question and its definition; do not substitute an assessment of risk level or severity. ")
    elif plan.diagnosis and len(evidence) >= 2 and evidence[0].get("is_focused"):
        scope = "Write one short claim naming the preferred test method from Source 1, then one short claim naming the recommended specimen for a living patient from Source 2. "
    elif plan.broad:
        scope = ("First define the subject in one plain sentence from Source 1. "
                 if evidence and evidence[0].get("is_definition") else "Define the subject if a passage supports it. ")
        scope += "Then cover 2-4 different central points relevant to the question; avoid repeating the same aspect. "
    elif re.search(r"\b(?:deadly|lethal|fatality|mortality|death rate)\b", question, re.I):
        scope = ("Give one plain-language mortality estimate from a broad summary that directly answers the question. "
                 "Prefer an overall average and range over a study-specific pooled estimate when both are provided. "
                 "Avoid study-design details and unrelated outbreaks. ")
    elif topic_request:
        scope = "Give a short overview of the named topic, covering main approaches or facts before details. "
        if len(plan.queries) > 1:
            scope += "Focus on the named aspect; do not substitute a general definition of the subject. "
    else:
        scope = "Answer the exact question first in one or two direct sentences; add a third only if needed. "
        if re.search(r"\b(?:diagnos|diagnosis|diagnosed)\w*\b", question, re.I):
            scope += "If the passages provide both a test method and a recommended specimen, include both. "
    sources = "\n\n".join(
        f"Source {number} ({item['source']}, page {item['page']}):\n{item['text']}"
        for number, item in enumerate(evidence, 1)
    )
    return (
        "You are a source-grounded document assistant for a general reader. Use only the passages below. "
        "Answer this question directly in everyday words. " + scope +
        f"Write at most {_claim_limit(question)} short, distinct factual claims. "
        "Each claim must be fully supported by ONE numbered passage and must address the question. "
        "Do not copy bibliography entries, document titles, study methods, or internal '(Source N)' markers as facts. "
        "Do not add doses, statistics, species taxonomy, or clinical details unless asked or essential to the answer. "
        "Do not infer safety, absence of risk, or a specific exposure route from general information. "
        "Finish each claim as a complete sentence; avoid long clauses and dangling words. "
        "Keep different diseases, subtypes, and patient groups separate. If the passages do not answer, return an empty list. "
        "Return JSON only: "
        '{"claims":[{"text":"One factual sentence without a citation","source_id":1}]}. '
        f"{feedback}\n\n{sources}\n\nAnswer this question directly: {question}"
    )


def _from_model_json(content: str, evidence: list[dict], max_claims: int = 4) -> str:
    data = json.loads(content)
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError("Answer model did not return a list of claims.")
    if not claims:
        return "I cannot find this in the source documents."
    sentences = []
    seen = set()
    for claim in claims[:max_claims]:
        if not isinstance(claim, dict) or type(claim.get("source_id")) is not int:
            raise ValueError("Answer model returned an invalid source ID.")
        source_id = claim["source_id"]
        if not 1 <= source_id <= len(evidence):
            raise ValueError("Answer model cited a passage outside the retrieved evidence.")
        statement = claim.get("text")
        if not isinstance(statement, str) or not statement.strip() or "[" in statement or "\n" in statement:
            raise ValueError("Answer model returned an invalid claim.")
        if re.search(r"(?:[,;:]|\b(?:a|an|the|and|or|but|to|of|through|with|via|from|including|because|by|for|in|on|as|at))\s*[.!?]?\s*$", statement, re.I):
            raise ValueError("Answer model returned an unfinished claim.")
        normalized = re.sub(r"\W+", " ", statement.casefold()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        item = evidence[source_id - 1]
        protected = re.sub(r"\b(?:e\.g|i\.e|vs|etc)\.",
                           lambda match: match.group(0).replace(".", "<DOT>"), statement, flags=re.I)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected):
            sentence = sentence.replace("<DOT>", ".").strip().rstrip(". ")
            if sentence:
                sentences.append(f"{sentence} [{item['source']}, page {item['page']}].")
    return "\n\n".join(sentences)


def compose_answer(question: str, evidence: list[dict], ollama_model: str = DEFAULT_MODEL, feedback: str = "") -> str:
    max_claims = _claim_limit(question)
    schema = {
        "type": "object",
        "properties": {"claims": {
            "type": "array", "maxItems": max_claims,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 420},
                    "source_id": {"type": "integer", "minimum": 1, "maximum": len(evidence)},
                },
                "required": ["text", "source_id"],
                "additionalProperties": False,
            },
        }},
        "required": ["claims"],
        "additionalProperties": False,
    }
    response = ollama.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": _make_prompt(question, evidence, feedback)}],
        format=schema,
        options={"temperature": 0, "num_predict": 768},
    )
    return _from_model_json(response["message"]["content"], evidence, max_claims)


def repair_answer(question: str, evidence: list[dict], rejected: list[dict], issues: list[str],
                  accepted: list[dict], ollama_model: str = DEFAULT_MODEL) -> str:
    """Make one bounded attempt to replace failed claims with supported ones."""
    rejected_text = "; ".join(str(item.get("text", "")) for item in rejected)[:900]
    accepted_text = "; ".join(str(item.get("text", "")) for item in accepted)[:900]
    feedback = (
        "The previous draft could not be fully checked. Rewrite only the missing or rejected facts. "
        "Use a shorter claim supported by one passage; omit a fact if no passage directly supports it. "
        "If a causal claim failed, state only what the source actually says about the activity and mechanism. "
        "Do not turn a related observation into a categorical claim. "
        "Do not repeat these accepted facts: " + accepted_text + "\n"
        "Rejected facts: " + rejected_text + "\n"
        "Check concerns: " + "; ".join(issues)[:700]
    )
    return compose_answer(question, evidence, ollama_model, feedback)
