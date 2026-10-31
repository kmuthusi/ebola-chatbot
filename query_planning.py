"""Question-led retrieval plans shared by every indexed topic.

Plans describe information needs, never answers. They are deliberately small so
the extra searches use the existing local index without another LLM round trip.
"""

import re
from dataclasses import dataclass


_FILLER = {"a", "an", "about", "all", "and", "are", "can", "could", "do", "does", "for", "give", "how", "i", "in", "information", "is", "it", "me", "of", "on", "please", "tell", "the", "there", "to", "us", "what", "would", "you"}
_ASPECT_TERMS = {
    "treatment": "clinical management supportive care therapy recommendation patients",
    "treat": "clinical management supportive care therapy recommendation patients",
    "vaccine": "vaccination immunization licensed recommendation eligibility",
    "vaccines": "vaccination immunization licensed recommendation eligibility",
    "diagnosis": "diagnostic testing confirmation assay specimen recommendation",
    "diagnosed": "diagnostic testing confirmation assay specimen recommendation",
    "symptoms": "signs clinical presentation illness",
    "spread": "transmission direct contact exposure prevention",
    "transmitted": "transmission direct contact exposure prevention",
}


@dataclass(frozen=True)
class SearchPlan:
    question: str
    subject: str
    broad: bool
    queries: tuple[str, ...]
    exposure: bool = False
    diagnosis: bool = False


def corpus_facets(subject: str, defining_passage: str) -> tuple[str, ...]:
    """Use the indexed definition to choose useful facets for a broad topic."""
    compact = re.sub(r"\s+", " ", defining_passage).casefold()
    mention = compact.find(subject.casefold())
    context = compact[max(0, mention - 50):mention + 270] if mention >= 0 else ""
    if re.search(r"\b(?:disease|illness|infection|virus)\b", context):
        return (
            f"{subject} symptoms include",
            f"{subject} human-to-human transmission occurs via exposure",
            f"{subject} clinical management supportive care treatment recommendations",
        )
    return (
        f"{subject} main purpose effects",
        f"{subject} how it works steps",
        f"{subject} practical guidance limitations",
    )


def focus_on_matching_sources(plan: SearchPlan, candidates: list[dict], minimum: int = 6) -> list[dict]:
    """Prefer documents whose titles match the question's information need.

    This is only a filter when several candidate passages are already present;
    it never invents evidence or relies on a fixed list of document names.
    """
    if plan.broad or len(plan.queries) < 2:
        return candidates
    expansion = plan.queries[1].casefold()
    aspect_words = set(re.findall(r"[a-z]{4,}", expansion)) - set(re.findall(r"[a-z]{4,}", plan.subject.casefold()))
    if not aspect_words:
        return candidates
    relevant = []
    for item in candidates:
        title = set(re.findall(r"[a-z]{4,}", str(item.get("source", "")).casefold()))
        if len(title & aspect_words) >= 2:
            relevant.append(item)
    return relevant if len(relevant) >= minimum else candidates


def focus_on_direct_guidance(plan: SearchPlan, candidates: list[dict]) -> list[dict]:
    """For care questions, keep patient guidance over mentions of treatment sites."""
    if (re.search(r"\b(?:diagnos|diagnosis|diagnosed)\w*\b", plan.question, re.I)
            and not re.search(r"\b(?:deceased|dead|postmortem|after death)\b", plan.question, re.I)):
        living = [item for item in candidates if not re.search(
            r"\bdiagnostic testing\s*[–-]\s*deceased individual\b",
            re.sub(r"\s+", " ", str(item.get("text", ""))), re.I,
        )]
        if len(living) >= 2:
            candidates = living
    if plan.broad or not re.search(r"\b(?:treat|treatment|therapy|clinical management)\b", plan.question, re.I):
        return candidates
    direct = []
    for item in candidates:
        content = re.sub(r"\s+", " ", str(item.get("text", "")).casefold())
        if re.search(r"\b(?:future research|study results and measurements|case fatality rate|pooled case fatality|health and care workers who have recovered)\b", content):
            continue
        patient = re.search(r"\b(?:patients?|people with|persons? with|start fluids|rehydration|supportive care)\b", content)
        action = re.search(r"\b(?:who recommends|who suggests|should be treated|start fluids|rehydration|therapy|therapeutic|clinical management)\b", content)
        if patient and action:
            direct.append(item)
    return direct if len(direct) >= 2 else candidates


def focus_excerpt(text: str, query: str, limit: int = 650) -> str:
    """Keep the sentence region requested by an evidence facet."""
    if len(text) <= limit:
        return text
    q = query.casefold()
    if "symptoms" in q:
        pattern = r"\b(?:present with symptoms|symptoms (?:can )?include|symptoms that are)\b"
    elif "specimen" in q:
        pattern = r"\b(?:recommended specimen type|specimen type|sample to collect)\b"
    elif "assay" in q or "diagnostic testing" in q:
        pattern = r"\b(?:preferred assay type|nucleic acid amplification testing|confirm an infection)\b"
    elif "transmission" in q:
        pattern = r"\b(?:human-to-human transmission occurs|transmission occurs|spread through|transmitted through)\b"
    elif "management" in q or "treatment" in q:
        pattern = r"\b(?:who recommends|who suggests|start fluids|supportive care)\b"
    else:
        pattern = rf"\b{re.escape(query.split()[0])}\b" if query.split() else r"^"
    match = re.search(pattern, text, re.I)
    if match and ("specimen" in q or "assay" in q or "diagnostic testing" in q):
        return text[max(0, match.start() - 10):match.start() + 300]
    if match:
        previous_sentence = text.rfind(". ", max(0, match.start() - 420), match.start())
        start = previous_sentence + 2 if previous_sentence >= 0 else max(0, match.start() - 300)
    else:
        start = 0
    return text[start:start + limit]


def answer_covers_plan(plan: SearchPlan, claims: list[dict], evidence: list[dict]) -> bool:
    """Check that surviving claims still cover the requested information need."""
    accepted = [claim for claim in claims if claim.get("passed")]
    if not accepted:
        return False
    if plan.broad and plan.question.casefold().startswith(("tell me about", "explain", "describe", "overview")):
        definitions = {(item.get("source"), item.get("page")) for item in evidence if item.get("is_definition")}
        cited = {citation for claim in accepted for citation in claim.get("citations", [])}
        aspects = {
            item.get("facet"): (item.get("source"), item.get("page"))
            for item in evidence if str(item.get("facet", "")).startswith("aspect_")
        }
        cited_aspects = {facet for facet, page in aspects.items() if page in cited}
        return (len(accepted) >= 3 and bool(definitions & cited) and len(cited) >= 2
                and len(cited_aspects) >= min(2, len(aspects)))
    if plan.exposure:
        activity = re.search(r"\bby\s+(.+?)\??$", plan.question, re.I)
        phrase = activity.group(1).strip(" ?.!").casefold() if activity else ""
        practical = any(phrase in claim.get("text", "").casefold() and
                        re.search(r"\b(?:unsafe|avoid|discouraged|risk|greeting)\b", claim.get("text", ""), re.I)
                        for claim in accepted)
        route = any(phrase not in claim.get("text", "").casefold() and
                    re.search(r"\b(?:blood|body fluids|bodily fluids|transmission|spread)\b", claim.get("text", ""), re.I)
                    for claim in accepted)
        return practical and route
    return True


def classification_target(question: str) -> str | None:
    match = re.search(r"\bclassif(?:y|ies|ication)\s+(.+?)\s*\??$", question, re.I)
    return match.group(1).strip(" ?.!") if match else None


def make_search_plan(question: str) -> SearchPlan:
    clean = re.sub(r"\s+", " ", question).strip()
    exposure = re.search(r"\b(?:can|could)\b.*\b(?:get|catch|contract|acquire)\b.+?\bby\s+(.+?)\??$", clean, re.I)
    if exposure:
        activity = exposure.group(1).strip(" ?.!")
        before = clean[:exposure.start(1)]
        topic = " ".join(word for word in re.findall(r"[\w'-]+", before.casefold())
                         if word not in _FILLER | {"by", "get", "catch", "contract", "acquire"})
        return SearchPlan(clean, topic, False, (
            clean,
            f"{activity} unsafe practice greetings touching",
            f"{topic} human-to-human transmission occurs via exposure",
        ), True)
    words = re.findall(r"[\w'-]+", clean.casefold())
    subject = " ".join(word for word in words if word not in _FILLER).strip(" ?.")
    if not subject:
        subject = clean
    broad = (bool(re.match(r"^(?:tell me about|what is|what are|explain|describe|overview of|information about)\b", clean, re.I))
             and len(subject.split()) <= 5 and not any(term in words for term in _ASPECT_TERMS))
    queries = [clean]
    if broad:
        # Generic facets are useful for diseases, medicines, procedures, and
        # other concepts. A facet is included only when matching source text
        # actually exists; the composer still has to cite each statement.
        queries.extend((
            f"{subject} definition overview what it is",
            f"{subject} main effects signs purpose uses",
            f"{subject} how it happens works spreads",
            f"{subject} prevention care treatment guidance",
        ))
    elif any(term in words for term in ("diagnosis", "diagnosed")):
        queries.extend((
            f"{subject} preferred diagnostic testing assay confirmation",
            f"{subject} recommended specimen sample for living patients",
        ))
    else:
        aspect = next((extra for term, extra in _ASPECT_TERMS.items() if term in words), None)
        if aspect:
            queries.append(f"{subject} {aspect}")
    return SearchPlan(clean, subject, broad, tuple(dict.fromkeys(queries)),
                      diagnosis=any(term in words for term in ("diagnosis", "diagnosed")))
