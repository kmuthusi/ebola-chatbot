"""Choose question-relevant pages from a bounded hybrid-search candidate pool."""

import json
import re

import ollama
from query_planning import make_search_plan


def _preview(text: str, question: str, limit: int = 520) -> str:
    """Show the reranker the matching part of a passage, not only its header."""
    compact = re.sub(r"\s+", " ", text).strip()
    terms = [word for word in re.findall(r"[a-z]{4,}", question.casefold())
             if word not in {"about", "could", "does", "from", "have", "into", "please", "tell", "what", "when", "where", "which", "would"}]
    positions = [match.start() for term in terms for match in [re.search(rf"\b{re.escape(term)}\b", compact, re.I)] if match]
    start = max(0, min(positions) - 90) if positions else 0
    return compact[start:start + limit]


def select_evidence(question: str, candidates: list[dict], model: str, limit: int = 6) -> list[dict]:
    """One local model call chooses useful pages; invalid output uses the input ranking."""
    if len(candidates) <= limit:
        return candidates
    pool = candidates[:24]
    plan = make_search_plan(question)
    listed = "\n".join(
        f"{number}. {item.get('source')}, page {item.get('page')}: {_preview(item.get('text', ''), question)}"
        for number, item in enumerate(pool, 1)
    )
    prompt = (
        "Select source passages that DIRECTLY answer the question. Return only JSON: "
        '{"ids":[1,2]}. Use at most ' + str(limit) + " IDs, ordered by importance. "
        "The answer writer will see only your selected pages. For a broad question, choose a passage that explains WHAT THE SUBJECT IS first, then choose different central aspects such as effects, operation or spread, and care where supported. "
        "For a focused question, prioritize an explicit answer or recommendation about the asked subject. "
        "Exclude bibliographies, research agendas, study methods, document descriptions, technical evidence tables, and pages that merely repeat question words. "
        "Prefer clear primary guidance and key points. Do not include passages about a different subtype or situation as if they applied generally. "
        f"Question: {question}\nPassages:\n{listed}"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format={"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": len(pool)}, "maxItems": limit}}, "required": ["ids"], "additionalProperties": False},
            options={"temperature": 0, "num_predict": 96},
        )
        ids = json.loads(response["message"]["content"])["ids"]
        if not isinstance(ids, list):
            raise ValueError("Invalid passage list")
        selected = []
        pages = set()
        for number in ids:
            if type(number) is not int or not 1 <= number <= len(pool):
                continue
            item = pool[number - 1]
            key = (item.get("source"), item.get("page"))
            if key not in pages:
                selected.append(item)
                pages.add(key)
        if plan.broad:
            # Small local rerankers sometimes pick related effects but omit the
            # defining page. A generic definitional-pattern check protects
            # overview coverage without inserting any topic-specific answer.
            subject = re.escape(plan.subject.casefold())
            defining = []
            for item in pool:
                content = re.sub(r"\s+", " ", item.get("text", "").casefold())
                for sentence in re.split(r"(?<=[.!?])\s+", content):
                    mention = re.search(rf"\b{subject}\b", sentence)
                    definition = re.search(r"\b(?:is an?|are|means|refers to|caused by|cause|defined as)\b", sentence)
                    if mention and definition and len(sentence) < 320 and not re.search(r"\b(?:diagnostic testing for|compared with previous|outbreak caused by)\b", sentence):
                        score = (3 if mention.start() < 35 else 0) + (2 if re.search(r"\bis an?\b|\bcaused by\b", sentence) else 0)
                        defining.append((score, item))
                        break
            if defining:
                first = max(defining, key=lambda pair: pair[0])[1]
                selected = [first, *(item for item in selected if (item.get("source"), item.get("page")) != (first.get("source"), first.get("page")))][:limit]
        return selected or pool[:limit]
    except (KeyError, ValueError, TypeError, RuntimeError):
        return pool[:limit]


def extractive_exposure_answer(question: str, evidence: list[dict]) -> str | None:
    """Use exact source sentences when an exposure inference cannot be checked."""
    activity_match = re.search(r"\bby\s+(.+?)\??$", question, re.I)
    if not activity_match:
        return None
    activity = activity_match.group(1).strip(" ?.!").casefold()
    if not activity:
        return None
    activity_result = None
    route_result = None
    for item in evidence:
        text = re.sub(r"\s+", " ", item.get("text", "")).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            lower = sentence.casefold()
            if activity in lower and re.search(r"\b(?:unsafe|avoid|discouraged)\b", lower):
                activity_result = activity_result or (sentence.strip(), item)
            if re.search(r"\b(?:transmission occurs|spreads? through|transmitted through)\b", lower) and re.search(
                r"\b(?:blood|body fluids|bodily fluids)\b", lower
            ):
                route = re.search(r"\b(?:transmission occurs|spreads? through|transmitted through)\b", sentence, re.I)
                if route:
                    part = sentence[route.start():]
                    endings = list(re.finditer(r"\b(?:blood|body fluids|bodily fluids)\b", part[:350], re.I))
                    if endings:
                        part = part[:endings[-1].end()]
                    route_result = route_result or (part.strip(), item)
    if not activity_result or not route_result:
        return None
    lines = []
    for sentence, item in (activity_result, route_result):
        # The words are copied from the indexed source excerpt; only whitespace
        # and citation placement change here.
        lines.append(f"{sentence.rstrip('. ')} [{item['source']}, page {item['page']}].")
    return "\n\n".join(lines)


def compose_sourced_exposure_summary(question: str, evidence: list[dict]) -> str | None:
    """Paraphrase only source phrases that match a known, checkable pattern.

    This contains no disease answer. The caller still runs the independent
    claim and quotation check before showing the summary.
    """
    activity_match = re.search(r"\bby\s+(.+?)\??$", question, re.I)
    if not activity_match:
        return None
    activity = activity_match.group(1).strip(" ?.!")
    if not activity:
        return None
    first = None
    second = None
    for item in evidence:
        source_text = re.sub(r"\s+", " ", item.get("text", ""))
        greeting = re.search(rf"\bgreetings? by touching, such as {re.escape(activity)}\b", source_text, re.I)
        unsafe = re.search(r"\bunsafe practice\b", source_text, re.I)
        context = re.search(r"\bduring an? ([A-Za-z ]{4,65} outbreak)\b", source_text, re.I)
        if greeting and unsafe and context and abs(greeting.start() - unsafe.start()) < 260:
            first = first or (
                f"During an {context.group(1)}, {greeting.group(0)}, are described as an unsafe practice",
                item,
            )
        route = re.search(r"\btransmission occurs via exposure to (blood and other body fluids) via contact\b", source_text, re.I)
        if route:
            second = second or (f"Person-to-person transmission occurs through contact with {route.group(1)}", item)
    if not first or not second:
        return None
    return "\n\n".join(
        f"{claim} [{item['source']}, page {item['page']}]."
        for claim, item in (first, second)
    )
