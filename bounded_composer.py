"""Compose source-bound answers from retrieved PDF passages."""

import json
import os
import re

import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def _claim_limit(question: str) -> int:
    """Reserve longer answers for broad or explicitly multi-part questions."""
    if re.match(r"\s*(?:how|which|can|could|is|are|does|do|what does|what should)\b", question, re.I) \
            and not re.search(r"\b(?:and|or)\b|\?.+\?", question, re.I):
        return 2
    return 4


def _make_prompt(question: str, evidence: list[dict], feedback: str = "") -> str:
    sources = "\n\n".join(
        f"Source {number} ({item['source']}, page {item['page']}):\n{item['text']}"
        for number, item in enumerate(evidence, 1)
    )
    return (
        "You answer questions about the supplied documents for a general reader. "
        "Use only the numbered source passages. Answer the question directly in everyday words, "
        f"using up to {_claim_limit(question)} short factual statements. Cover each distinct part of the question that the passages answer. "
        "Start with the exact information requested. Usually one or two statements are enough. "
        "Do not add background facts, statistics, or related advice unless the question asks for them. "
        "When asked for extra requirements, differences, or exceptions, state the exceptions directly instead of "
        "listing the shared baseline. "
        "For causal or safety questions, distinguish a described activity from a proven mechanism; "
        "do not infer categorical safety or harm from related facts. "
        "If separate passages describe an activity and its mechanism, cite them in separate claims. "
        "For a classification question, include the named parent category from the source rather than only a generic description. "
        "When asked how something happens, explain the mechanism directly. "
        "For a question about a method, name the method when the passages provide it instead of describing only its purpose. "
        "Keep a statement only if its chosen source "
        "directly supports the entire statement. Split facts that need different sources. "
        "Check that the source discusses the same named entity and subject as the statement; "
        "a passage about a different subject cannot support it. "
        "Do not add examples that are absent from the chosen passage. "
        "Do not add a heading, introduction, recommendation, or detail the question did not ask for. "
        "Keep different entities, populations, dates, and conditions separate. "
        "Return JSON only with this shape: "
        '{"claims":[{"text":"One factual sentence without a citation","source_id":1}]}. '
        "Use the number of the passage that supports each claim. If the passages do not answer "
        "the question, return {\"claims\":[]}. "
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
        normalized = re.sub(r"\W+", " ", statement.casefold()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        item = evidence[source_id - 1]
        sentences.append(f"{statement.strip().rstrip('. ')} [{item['source']}, page {item['page']}].")
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
                    "text": {"type": "string", "maxLength": 240},
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
        options={"temperature": 0, "num_predict": 512},
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
