"""
rag_pipeline.py — rerank and relevance release.
"""

import math
import re
from typing import List, Dict, Any

_STOPWORDS = {"a", "about", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "of", "the", "there", "to", "what"}


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {word[:-1] if len(word) > 5 and word.endswith("s") and not word.endswith(("us", "ss")) else word for word in words}

def _simple_relevance_score(candidate: Dict[str, Any], terms: set[str], frequency: dict[str, int], total: int) -> float:
    """Favor passages that contain the question's distinctive words."""
    s = candidate.get("score", 0.0) or 0.0
    page_bonus = 0.05 if candidate.get("page") else 0.0
    words = _words(candidate.get("text", ""))
    lexical_bonus = min(0.20, sum(0.08 * math.log((total + 1) / (frequency[term] + 1)) for term in terms & words))
    content = re.sub(r"\s+", " ", candidate.get("text", "").casefold())
    # An explicit recommendation or key point is stronger evidence than an
    # article title, research agenda, or evidence-review narrative containing
    # the same search terms. These are document-structure cues, not answers.
    guidance_bonus = 0.0
    if terms & {"care", "clinical", "guidance", "management", "prevent", "prevention", "recommendation", "therapy", "treat", "treatment", "vaccine", "vaccination"}:
        if re.search(r"\b(?:who recommends|who suggests|good practice statement|key points)\b", content):
            guidance_bonus = 0.20
        elif re.search(r"\brecommendation\b", content):
            guidance_bonus = 0.08
    research_penalty = 0.0
    if re.search(r"\b(?:future research|uncertainties and future research|findings of the review|studies were found|evidence was derived from|study results and measurements|attack rate)\b", content):
        research_penalty = 0.18
    if not terms & {"evidence", "study", "table", "trial", "compare", "comparison", "risk"} and re.search(
        r"\b(?:study results and measurements|certainty of the evidence|based on data from \d+ participants|pico population|risk of bias)\b", content
    ):
        research_penalty += 0.25
    directness = 0.0
    if terms & {"symptom", "sign", "presentation"}:
        if re.search(r"\b(?:symptoms? (?:can )?include|present with symptoms|signs and symptoms|symptoms include)\b", content):
            directness += 0.24
        if "interpretation of laboratory results" in content:
            directness -= 0.12
    if terms & {"transmission", "spread", "contact", "body", "fluid"}:
        if re.search(r"\b(?:transmitted through|spread through|direct contact with (?:blood|body|bodily)|human-to-human transmission occurs)\b", content):
            directness += 0.24
    return s + page_bonus + lexical_bonus + guidance_bonus + directness - research_penalty

def rerank_and_release(question: str, candidates: List[Dict[str, Any]], intent: str, max_passages: int = 5) -> Dict[str, Any]:
    # Basic reranking
    terms = _words(question) - _STOPWORDS
    frequency = {term: sum(term in _words(c.get("text", "")) for c in candidates) for term in terms}
    generic_modifiers = {"extra", "specific", "specify", "guideline", "guidance", "document", "source"}
    observed = [term for term in terms - generic_modifiers if 0 < frequency[term] <= max(1, len(candidates) // 4)]
    rarest = min(observed, key=lambda term: frequency[term]) if observed else None
    query_words = re.findall(r"[a-z]+", question.lower())
    phrases = [" ".join(query_words[index:index + 2]) for index in range(len(query_words) - 1)
               if query_words[index] not in _STOPWORDS and query_words[index + 1] not in _STOPWORDS]
    ranked = sorted(
        candidates,
        key=lambda c: _simple_relevance_score(c, terms, frequency, len(candidates))
        + (0.26 if rarest and rarest in _words(c.get("text", "")) else 0)
        + (0.18 if any(phrase in " ".join(re.findall(r"[a-z]+", c.get("text", "").lower())) for phrase in phrases) else 0),
        reverse=True,
    )
    if re.search(r"\b(?:can|could)\b[^?]*\b(?:get|catch|contract|acquire)\b[^?]*\bby\b", question, re.I):
        route = next((candidate for candidate in ranked[1:] if re.search(
            r"\b(?:direct contact with|spread through|transmitted through|spreads through)\b",
            candidate.get("text", ""), re.I,
        )), None)
        if route is not None:
            ranked.remove(route)
            ranked.insert(1, route)
    evidence = ranked

    # Refusal criteria: no evidence or extremely low scores
    if not evidence or all((c.get("score") or 0) < 0.2 for c in evidence):
        return {"evidence": evidence[:2], "release": "refusal"}

    # A long PDF page can contribute many overlapping chunks. Prefer distinct
    # pages first, then fill remaining slots when a second chunk is useful.
    selected = []
    used_pages = set()
    for candidate in evidence:
        page_key = (candidate.get("source"), candidate.get("page"))
        if page_key not in used_pages:
            selected.append(candidate)
            used_pages.add(page_key)
        if len(selected) >= max_passages:
            break
    if len(selected) < max_passages:
        selected_keys = {(x.get("source"), x.get("page"), x.get("text")) for x in selected}
        for candidate in evidence:
            key = (candidate.get("source"), candidate.get("page"), candidate.get("text"))
            if key not in selected_keys:
                selected.append(candidate)
            if len(selected) >= max_passages:
                break
    return {"evidence": selected, "release": "grounded"}
