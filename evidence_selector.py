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


def extractive_answer(question: str, evidence: list[dict], facet_queries: tuple[str, ...]) -> str | None:
    """Copy relevant source sentences when the answer model cannot ground a draft.

    This fallback contains no topic facts or answer templates. Every returned
    sentence comes from a retrieved passage and is checked by the caller.
    """
    if not evidence or not facet_queries:
        return None
    lines = []
    used = set()
    stop = {"about", "after", "before", "could", "does", "from", "have", "into",
            "other", "their", "there", "these", "those", "through", "which", "would"}
    for query, item in zip(facet_queries, evidence):
        terms = {word for word in re.findall(r"[a-z]{4,}", query.casefold()) if word not in stop}
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", item.get("text", "")).strip())
        ranked = []
        for sentence in sentences:
            sentence = sentence.strip()
            words = set(re.findall(r"[a-z]{4,}", sentence.casefold()))
            overlap = len(terms & words)
            if 35 <= len(sentence) <= 500 and overlap >= 2 and (item.get("source"), item.get("page"), sentence) not in used:
                ranked.append((overlap, -len(sentence), sentence))
        if not ranked:
            return None
        sentence = max(ranked)[2]
        used.add((item.get("source"), item.get("page"), sentence))
        lines.append(f"{sentence.rstrip('. ')} [{item['source']}, page {item['page']}].")
    return "\n\n".join(lines)
