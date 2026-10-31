"""Conservative, source-bound checks of generated factual claims."""

import json
import re
import unicodedata
from difflib import SequenceMatcher

import ollama

from answer_formatting import format_answer_markdown
from verifier import CITATION_RE


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def _quote_matches(quote: str, excerpts: list[str]) -> bool:
    if re.search(r"\.\.\.|…", quote):
        parts = [_normalize(part) for part in re.split(r"\.\.\.|…", quote) if len(_normalize(part)) >= 12]
        if len(parts) < 2:
            return False
        for excerpt in excerpts:
            source = _normalize(excerpt)
            position = 0
            for part in parts:
                found = source.find(part, position)
                if found < 0:
                    break
                position = found + len(part)
            else:
                return True
        return False
    normalized = _normalize(quote)
    if len(normalized) < 12:
        return False
    for excerpt in excerpts:
        source = _normalize(excerpt)
        if normalized in source:
            return True
        matches = [match for match in SequenceMatcher(None, normalized, source, autojunk=False).get_matching_blocks() if match.size]
        matched = sum(match.size for match in matches)
        source_span = matches[-1].b + matches[-1].size - matches[0].b if matches else 0
        if matched / len(normalized) >= 0.92 and source_span <= max(len(normalized) * 1.25, len(normalized) + 50):
            return True
    return False


def _unquoted_examples(claim: str, quote: str) -> list[str]:
    """Find concrete examples the cited quote does not actually name."""
    match = re.search(r"\b(?:such as|including|for example|like)\s+([^.;]+)", claim, re.I)
    if not match:
        return []
    quoted = _normalize(quote)
    example_text = re.split(r",\s+as\s+(?:an?\s+)?(?:unsafe|safe|important|common)\b", match.group(1), maxsplit=1, flags=re.I)[0]
    examples = re.split(r",|\b(?:and|or)\b", example_text, flags=re.I)
    missing = []
    for example in examples:
        example = re.split(
            r"\b(?:is|are|was|were|should|can|may|must|has|have|does|do|did|will|would|for|in|during|among)\b",
            example,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        normalized = _normalize(re.sub(r"^(?:a|an|the)\s+", "", example.strip(), flags=re.I))
        if len(normalized) >= 3 and normalized not in quoted:
            missing.append(normalized)
    return missing


def _unquoted_list_items(claim: str, quote: str) -> list[str]:
    """Require the supporting quote to cover every item in a claimed list."""
    match = re.search(r"\b(?:wear|include|includes|contain|contains|use|uses|consist of)\s+(.+)", claim, re.I)
    if not match or match.group(1).count(",") < 2:
        return []
    quote_words = set(_normalize(quote).split())
    missing = []
    for part in re.split(r",|\b(?:and|or)\b", match.group(1), flags=re.I):
        part = re.split(r"\b(?:during|when|while)\b", part, maxsplit=1, flags=re.I)[0]
        words = [word for word in _normalize(part).split()
                 if word not in {"a", "an", "the", "of", "in", "with", "for", "to", "by", "and", "or"}]
        if words and any(word not in quote_words for word in words):
            missing.append(" ".join(words))
    return missing


def _is_unrequested_statistic(question: str, claim: str) -> bool:
    """An unrelated metric aside is not part of a how/which answer."""
    return bool(
        re.match(r"\s*(?:how|which)\b", question, re.I)
        and not re.search(r"\b(?:rate|fatality|mortality|percent|percentage|count|total|score|revenue|cost|how many)\b", question, re.I)
        and re.search(r"\b(?:fatality|mortality|CFR|revenue|cost|rate|score)\b", claim, re.I)
    )


def _is_unrequested_definition(question: str, claim: str) -> bool:
    """A general definition does not explain a requested process or adaptation."""
    return bool(
        (re.match(r"\s*how\b", question, re.I)
         or re.search(r"\b(?:treatment|therapy|diagnosis|symptoms|vaccine|vaccination)\b", question, re.I))
        and not re.search(r"\b(?:defined|classified|classification)\b", question, re.I)
        and re.search(r"\b(?:is|are)\s+(?:a|an)\s+[^.;]{0,100}\b", claim, re.I)
        and not _strong_topic_match(question, claim)
    )


def _is_unrequested_detail(question: str, claim: str) -> bool:
    """Reject clinical detail or adjacent tasks outside a broad information need."""
    q = question.casefold()
    c = claim.casefold()
    asks_amount = bool(re.search(r"\b(?:dose|dosage|how much|how many|volume|amount|ml|mg|kg|regimen)\b", q))
    if not asks_amount and re.search(r"\b\d+(?:\.\d+)?\s*(?:ml|mg|g|mcg|µg)\b", c):
        return True
    if re.search(r"\b(?:treat|treatment|therapy|clinical management)\b", q):
        if re.search(r"\b(?:swabb?ed|surface|environmental audit)\b", c) and re.search(r"\b(?:treatment unit|treatment cent(?:er|re))\b", c):
            return True
        if re.search(r"\b(?:testing|specimen|assay|diagnos)\w*\b", c) and not re.search(
            r"\b(?:treat|treatment|therapy|therapeutic|rehydrat|fluids?|antiviral|antibiotic|supportive care)\b", c
        ):
            return True
    if not re.search(r"\b(?:which guideline|where can|what document|source)\b", q) and re.search(
        r"\b(?:guideline|document|manual)\b.{0,90}\b(?:provides details|describes|covers information)\b", c
    ):
        return True
    return False


def _is_category_error(claim: str) -> bool:
    """Keep a disease or condition distinct from the agent causing it."""
    return bool(re.search(
        r"\b(?:disease|illness|syndrome|condition)\s+(?:is|are|was|were)\s+(?:an?\s+)?(?:zoonotic\s+|infectious\s+)?(?:virus|pathogen|bacterium|parasite)\b",
        claim, re.I,
    ))


def _strong_topic_match(question: str, claim: str) -> bool:
    """Do not discard a direct answer because the model marked it background."""
    generic = {"about", "which", "what", "when", "where", "there", "should", "would", "could",
               "guideline", "guidance", "patients", "patient", "recommend", "recommended"}
    focus = {word[:5] for word in _normalize(question).split() if len(word) >= 5 and word not in generic}
    words = {word[:5] for word in _normalize(claim).split() if len(word) >= 5}
    return len(focus & words) >= 2


def _exposure_route_supported(claim: str, quote: str) -> bool:
    """A label for an activity alone does not prove a claimed outcome."""
    match = re.search(r"\bcan\s+(?:get|catch|contract|acquire)\b[^.;]*?\bby\s+([^,.;]+)", claim, re.I)
    if not match:
        return True
    activity = re.split(r"\b(?:with|if|when|according|during)\b", match.group(1), maxsplit=1, flags=re.I)[0]
    activity_words = re.findall(r"[a-z]+", activity.casefold())[:2]
    quote_words = _normalize(quote)
    activity_named = bool(activity_words and " ".join(activity_words) in quote_words)
    route_stated = bool(re.search(
        r"\b(?:spread|transmi\w*|infect\w*|expos\w*|acquir\w*|contract\w*|risk|"
        r"receiv\w*|obtain\w*|eligib\w*|qualif\w*|provid\w*|allow\w*|entitl\w*|caus\w*|result\w*)\b",
        quote, re.I,
    ))
    return activity_named and route_stated


def _specific_exposure_supported(question: str, claim: str, quote: str) -> bool:
    """A general mechanism cannot prove a named activity causes an outcome."""
    match = re.search(r"\b(?:can|could)\b[^?]*\b(?:get|catch|contract|acquire)\b[^?]*\bby\s+([^?]+)", question, re.I)
    if not match:
        return True
    activity = re.findall(r"[a-z]+", match.group(1).casefold())[:2]
    if not activity or " ".join(activity) not in _normalize(claim):
        return True
    if not re.search(r"\b(?:get|catch|contract|acquire|infect\w*|transmi\w*|spread|receiv\w*|obtain\w*)\b", claim, re.I):
        return True
    return bool(
        " ".join(activity) in _normalize(quote)
        and re.search(r"\b(?:spread|transmi\w*|infect\w*|expos\w*|acquir\w*|contract\w*|"
                      r"receiv\w*|obtain\w*|eligib\w*|qualif\w*|provid\w*|allow\w*|entitl\w*|caus\w*|result\w*)\b", quote, re.I)
    )


def split_claims(answer: str) -> list[str]:
    """Split prose and list items so a citation cannot cover unrelated sentences."""
    claims = []
    for raw_line in format_answer_markdown(answer).splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", raw_line)
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line or line.endswith(":"):
            continue
        safe_line = re.sub(
            r"\b(?:e\.g|i\.e|vs|etc)\.",
            lambda match: match.group(0).replace(".", "<DOT>"),
            line,
            flags=re.I,
        )
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", safe_line):
            sentence = sentence.replace("<DOT>", ".")
            content = CITATION_RE.sub("", sentence).strip(" .;:!?")
            if re.search(r"[A-Za-z]{2,}", content):
                claims.append(sentence.strip())
    return claims


def check_claims(answer: str, evidence: list[dict], model: str, question: str = "") -> dict:
    """Require a source citation and a matching quote for every supported claim.

    A model judges meaning, so a pass is an automated screening result, not
    clinical validation or proof of entailment.
    """
    claims = split_claims(answer)
    if not claims:
        return {"passed": False, "issues": ["No checkable factual claims found."], "claims": []}

    available = {}
    for item in evidence:
        key = (item.get("source"), item.get("page"))
        available.setdefault(key, []).append(item.get("text", ""))

    prepared = []
    issues = []
    for number, claim in enumerate(claims, 1):
        citations = [(match.group(1), int(match.group(2))) for match in CITATION_RE.finditer(claim)]
        if not citations:
            issues.append(f"Claim {number} has no document and page citation.")
        invalid = [citation for citation in citations if citation not in available]
        if invalid:
            issues.append(f"Claim {number} cites pages outside the retrieved evidence: {invalid}.")
        prepared.append({"id": number, "text": CITATION_RE.sub("", claim).strip(), "original": claim, "citations": citations})
    if issues:
        return {"passed": False, "issues": issues, "claims": prepared}

    review_items = []
    question_names = {
        term.casefold() for term in re.findall(r"\b(?:[A-Z][a-z]{3,}|[A-Z]{2,})\b", question)
        if term.casefold() not in {"are", "can", "does", "how", "is", "please", "tell", "the", "what", "when", "where", "which", "who", "ipc", "who"}
    }
    for claim in prepared:
        excerpts = [
            {"source": source, "page": page, "text": "\n".join(available[(source, page)])}
            for source, page in claim["citations"]
        ]
        review_items.append({"id": claim["id"], "claim": claim["text"], "cited_excerpts": excerpts})

    instructions = (
        "You are checking factual claims against cited PDF excerpts. Use ONLY the text in each "
        "claim's cited_excerpts. Ignore outside knowledge and the citation label as evidence. "
        "For each claim, return its original id and verdict 'supported', 'unsupported', or 'unclear'. "
        "Use 'supported' only when the cited text directly supports the WHOLE claim, including "
        "the named entity, population, timing, causal route, qualifiers, numbers, and recommendations. "
        "A relevant topic or related fact is not enough. Accept faithful paraphrases, but do not supply missing details from general knowledge. "
        "Do not generalize a statement about one subtype, outbreak, population, or situation to the whole parent category. "
        "For relevant=true, the claim must answer the user's actual information need, not merely contain a word from the question. "
        "A bibliography entry, a pointer to another guideline, a description of a study or facility, or a fact about a different subject is not a direct answer. "
        "For a broad overview, a definition and central features are relevant; incidental locations, document descriptions, and study methods are not. "
        "For supported claims, provide an exact "
        "verbatim quote copied from a cited excerpt that conveys the supporting fact. "
        "A claim that something cannot happen, is safe, or does not spread needs an explicit statement about that same route or condition. "
        "Calling an activity unsafe is not by itself evidence that it always transmits infection. "
        "Do not infer a negative claim merely because a source lists other routes or conditions. "
        "Every concrete example introduced by 'such as', 'including', or 'like' must be named in the cited excerpt. "
        "Mark relevant=false for a claim that only adds background, statistics, or another topic beyond what the question asks. "
        "Also decide whether the answer as a whole directly addresses the question using only the cited excerpts. "
        "If it gives related facts but misses the requested recommendation, classification, condition, or mechanism, "
        "set addresses_question=false and name the missing aspect briefly. "
        "For other verdicts, provide a brief reason. Return JSON only: "
        '{"claims":[{"id":1,"verdict":"supported","quote":"exact source text","reason":"","relevant":true}],'
        '"addresses_question":true,"missing_aspects":""}. '
    )
    prompt = instructions + f"Question: {question}\nAnswer: {answer}\nReview items: {json.dumps(review_items, ensure_ascii=False)}"
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        review_result = json.loads(response["message"]["content"])
        reviewed = review_result["claims"]
    except Exception as exc:
        return {
            "passed": False,
            "issues": [f"Automated claim check could not complete: {type(exc).__name__}."],
            "claims": prepared,
        }

    if not isinstance(reviewed, list) or len(reviewed) != len(prepared):
        # Small local models sometimes omit an item in a batch. Check each
        # claim independently before withholding the entire answer.
        reviewed = []
        review_result = {}
        for item in review_items:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": instructions + f"Review this one item: {json.dumps(item, ensure_ascii=False)}"}],
                    format="json",
                    options={"temperature": 0},
                )
                single = json.loads(response["message"]["content"])["claims"]
                if not isinstance(single, list) or len(single) != 1:
                    raise ValueError("Incomplete single-claim review")
                reviewed.append(single[0])
            except Exception:
                return {"passed": False, "issues": ["Automated claim check returned an incomplete result."], "claims": prepared}

    by_id = {}
    for item in reviewed:
        if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] in by_id:
            return {"passed": False, "issues": ["Automated claim check returned invalid claim IDs."], "claims": prepared}
        by_id[item["id"]] = item
    if set(by_id) != {claim["id"] for claim in prepared}:
        return {"passed": False, "issues": ["Automated claim check omitted a claim."], "claims": prepared}

    for claim in prepared:
        item = by_id[claim["id"]]
        verdict = item.get("verdict")
        quote = item.get("quote", "")
        claim["verdict"] = verdict
        claim["quote"] = quote if isinstance(quote, str) else ""
        claim["reason"] = str(item.get("reason", ""))[:300]
        claim["relevant"] = (
            (item.get("relevant", True) is not False or _strong_topic_match(question, claim["text"]))
            and not _is_unrequested_statistic(question, claim["text"])
            and not _is_unrequested_definition(question, claim["text"])
            and not _is_unrequested_detail(question, claim["text"])
            and not _is_category_error(claim["text"])
        )
        excerpts = [text for citation in claim["citations"] for text in available[citation]]
        quote_matches = isinstance(quote, str) and _quote_matches(quote, excerpts)
        negative_claim = bool(re.search(r"\b(?:no|not|never|cannot|safe|doesn't|isn't|aren't|won't)\b", claim["text"], re.I))
        explicit_exclusion = bool(isinstance(quote, str) and re.search(
            r"\b(?:no|not|never|cannot|only|safe|doesn't|isn't|aren't|won't)\b", quote, re.I,
        ))
        # A short verbatim quote can establish the core claim while a second
        # sentence in the same cited passage names its examples. Check the
        # full cited passage rather than rejecting that valid support.
        cited_text = " ".join(excerpts)
        missing_examples = _unquoted_examples(claim["text"], cited_text)
        missing_list_items = _unquoted_list_items(claim["text"], cited_text)
        exposure_route_supported = _exposure_route_supported(claim["text"], quote if isinstance(quote, str) else "")
        specific_exposure_supported = _specific_exposure_supported(question, claim["text"], quote if isinstance(quote, str) else "")
        named_terms = {
            term for term in re.findall(r"\b(?:[A-Z][a-z]{3,}|[A-Z]{2,})\b", claim["text"])
            if term.casefold() in question_names
        }
        missing_terms = [
            term for term in named_terms
            if not any(re.search(rf"\b{re.escape(term)}\b", excerpt, re.I) for excerpt in excerpts)
            and any(
                re.search(rf"\b{re.escape(other)}\b", excerpt, re.I)
                for other in question_names - {term.casefold()} for excerpt in excerpts
            )
        ]
        claim["passed"] = claim["relevant"] and verdict == "supported" and quote_matches and not missing_terms and not missing_examples and not missing_list_items and exposure_route_supported and specific_exposure_supported and (not negative_claim or explicit_exclusion)
        if not claim["relevant"]:
            continue
        if verdict != "supported":
            issues.append(f"Claim {claim['id']} was {verdict or 'not classified'} by the automated check.")
        elif not quote_matches:
            issues.append(f"Claim {claim['id']} has no matching verbatim source quote.")
        elif missing_terms:
            issues.append(f"Claim {claim['id']} names {', '.join(sorted(missing_terms))}, which is absent from its cited excerpt.")
        elif negative_claim and not explicit_exclusion:
            issues.append(f"Claim {claim['id']} makes a negative or safety claim without an explicit supporting exclusion.")
        elif missing_examples:
            issues.append(f"Claim {claim['id']} adds examples absent from the supporting quote: {', '.join(missing_examples)}.")
        elif missing_list_items:
            issues.append(f"Claim {claim['id']} lists items absent from its supporting quote: {', '.join(missing_list_items)}.")
        elif not exposure_route_supported:
            issues.append(f"Claim {claim['id']} asserts transmission through an activity without explicit route evidence.")
        elif not specific_exposure_supported:
            issues.append(f"Claim {claim['id']} applies a general transmission route to a named activity without direct evidence.")

    coverage = review_result.get("addresses_question", True) is not False
    missing_aspects = str(review_result.get("missing_aspects", ""))[:250]
    if not coverage:
        issues.append("The answer does not directly address the question" +
                      (f": {missing_aspects}." if missing_aspects else "."))
    if not any(claim.get("passed") for claim in prepared) and not issues:
        issues.append("No relevant supported claims answer the question.")
    return {"passed": not issues, "issues": issues, "claims": prepared,
            "addresses_question": coverage, "missing_aspects": missing_aspects}
