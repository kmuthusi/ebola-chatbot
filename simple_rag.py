"""Local RAG over all PDFs in a source directory."""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

from retrievers import DEFAULT_DATA_DIR, PDFDirectoryRetriever
from vector_store import VectorStore
from rag_pipeline import rerank_and_release
from query_planning import make_search_plan, corpus_facets, focus_on_matching_sources, focus_on_direct_guidance, focus_excerpt, answer_covers_plan, classification_target
from evidence_selector import select_evidence, extractive_exposure_answer, compose_sourced_exposure_summary
from bounded_composer import compose_answer, repair_answer
from verifier import verify_answer
from claim_checker import check_claims

DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def question_with_context(question: str, previous_question: str | None) -> str:
    """Turn a short follow-up into a standalone question for retrieval."""
    if not previous_question:
        return question
    topic = re.search(r"\b(?:about|on)\s+(.+?)\s*[?.!]*$", previous_question, re.I)
    subject = topic.group(1).strip(" ?.!") if topic else ""
    if not subject:
        return question
    if re.search(r"\b(?:it|its|they|them|their|this|that|those)\b", question, re.I):
        return re.sub(r"\b(?:it|this|that)\b", subject, question, flags=re.I)
    if re.match(r"\s*(?:how|what)\s+about\b", question, re.I):
        return f"{question.rstrip(' ?.')} in relation to {subject}?"
    return question


@dataclass
class SimpleAssistantRequest:
    question: str
    top_k: int = 6
    ollama_model: str = DEFAULT_MODEL
    previous_question: str | None = None


class SimpleAssistantRAG:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, ollama_model: str = DEFAULT_MODEL):
        self.data_dir = data_dir
        self.ollama_model = ollama_model
        self.sources = PDFDirectoryRetriever(data_dir)
        self.store = VectorStore()

    def index(self, fingerprint: str | None = None) -> None:
        self.store.index_chunks(self.sources.read_and_chunk())
        self.store.mark_source(fingerprint or self.sources.fingerprint())

    def ask(self, request: SimpleAssistantRequest) -> dict[str, Any]:
        effective_question = question_with_context(request.question, request.previous_question)
        fingerprint = self.sources.fingerprint()
        if not self.store.source_is_current(fingerprint):
            self.index(fingerprint)
        plan = make_search_plan(effective_question)
        candidates = self.store.search(effective_question, k=max(request.top_k, 25))
        anchors = []
        definition_hit = None
        if plan.broad and hasattr(self.store, "definition_search"):
            definitions = self.store.definition_search(plan.subject, k=1)
            if definitions:
                definition_hit = {**definitions[0], "is_definition": True, "facet": "definition"}
                candidates.append(definition_hit)
                anchors.append(definition_hit)
        facet_queries = corpus_facets(plan.subject, definition_hit["text"]) if plan.broad and definition_hit else plan.queries[1:]
        if facet_queries:
            seen = {(hit.get("source"), hit.get("page"), hit.get("text")) for hit in candidates}
            for facet_index, facet_query in enumerate(facet_queries):
                facet_hits = self.store.search(facet_query, k=18)
                facet_ranked = rerank_and_release(facet_query, facet_hits, intent="documents", max_passages=2)
                if facet_ranked["release"] == "grounded":
                    for hit in facet_ranked["evidence"]:
                        key = (hit.get("source"), hit.get("page"), hit.get("text"))
                        if key not in seen:
                            candidates.append(hit)
                            seen.add(key)
                    anchor = facet_ranked["evidence"][0]
                    if (plan.broad or plan.exposure or plan.diagnosis) and hasattr(self.store, "expand_evidence"):
                        expanded_anchor = self.store.expand_evidence([anchor], question=facet_query)[0]
                        anchor = {**expanded_anchor, "text": focus_excerpt(expanded_anchor["text"], facet_query), "is_focused": True}
                    anchors.append({**anchor, "facet": f"aspect_{facet_index}"})
        original_sources = {hit.get("source") for hit in candidates}
        candidates = focus_on_matching_sources(plan, candidates)
        candidates = focus_on_direct_guidance(plan, candidates)
        focused_sources = {hit.get("source") for hit in candidates} if len({hit.get("source") for hit in candidates}) < len(original_sources) else None
        anchors = [hit for hit in anchors if any(
            item.get("source") == hit.get("source") and item.get("page") == hit.get("page")
            for item in candidates
        )]
        pipeline = rerank_and_release(
            effective_question, candidates, intent="documents", max_passages=max(1, min(request.top_k, 8)),
        )
        evidence = pipeline["evidence"]
        if plan.diagnosis and len(anchors) >= 2 and pipeline["release"] == "grounded":
            evidence = anchors[:2]
        elif (plan.broad and len(anchors) >= 3 or plan.exposure and len(anchors) >= 2) and pipeline["release"] == "grounded":
            evidence = []
            used_pages = set()
            for hit in anchors:
                page = (hit.get("source"), hit.get("page"))
                if page not in used_pages:
                    evidence.append(hit)
                    used_pages.add(page)
                if len(evidence) >= max(1, min(request.top_k, 8)):
                    break
        elif pipeline["release"] == "grounded" and len(candidates) > 10:
            # The local selector sees compact previews of a bounded set of
            # distinct pages. It can keep definition, mechanism, and guidance
            # coverage without letting a keyword match decide the answer.
            ranked_pool = rerank_and_release(
                effective_question, candidates, intent="documents", max_passages=24,
            )["evidence"]
            pool = []
            used_pages = set()
            for hit in [*anchors, *ranked_pool]:
                page = (hit.get("source"), hit.get("page"))
                if page not in used_pages:
                    pool.append(hit)
                    used_pages.add(page)
                if len(pool) >= 24:
                    break
            evidence = select_evidence(effective_question, pool, request.ollama_model,
                                       limit=max(1, min(request.top_k, 8)))
        if definition_hit is not None and pipeline["release"] == "grounded":
            evidence = [definition_hit, *(item for item in evidence if
                         (item.get("source"), item.get("page")) !=
                         (definition_hit.get("source"), definition_hit.get("page")))][:max(1, min(request.top_k, 8))]
        if pipeline["release"] == "refusal":
            return {"answer": "I cannot find this in the source documents.", "status": "insufficient_context", "chunks": evidence, "issues": [], "claim_checks": []}
        if hasattr(self.store, "expand_evidence"):
            evidence = self.store.expand_evidence(evidence, question=effective_question)
        try:
            answer = compose_answer(effective_question, evidence, request.ollama_model)
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "answer": "I could not confirm this answer from the documents. Please check the source pages below or ask a more specific question.",
                "status": "review_required", "issues": [f"Answer generation failed: {exc}"],
                "claim_checks": [], "chunks": evidence,
            }
        if answer.strip() == "I cannot find this in the source documents.":
            return {"answer": answer, "status": "insufficient_context", "issues": [], "claim_checks": [], "chunks": evidence}
        verified = verify_answer(answer, evidence)
        if verified["issues"]:
            return {
                "answer": "I could not confirm this answer from the documents. Please check the source pages below or ask a more specific question.",
                "status": "review_required", "issues": verified["issues"], "claim_checks": [], "chunks": evidence,
            }
        checked = check_claims(answer, evidence, request.ollama_model, effective_question)
        if checked["passed"] and (plan.broad or plan.exposure) and not answer_covers_plan(plan, checked["claims"], evidence):
            checked = {**checked, "passed": False,
                       "issues": ["The checked claims do not cover the requested parts of the question."],
                       "addresses_question": False, "missing_aspects": "requested parts"}
        if not checked["passed"]:
            target = classification_target(effective_question)
            if target and hasattr(self.store, "definition_search"):
                definitions = self.store.definition_search(target, k=8)
                if definitions:
                    question_terms = set(re.findall(r"[a-z]{4,}", effective_question.casefold()))
                    chosen = max(definitions, key=lambda item: len(
                        question_terms & set(re.findall(r"[a-z]{4,}", item["source"].casefold()))
                    ))
                    source_statement = chosen["text"].strip().rstrip(". ")
                    statement = re.sub(r"\s+([,.;:])", r"\1", source_statement)
                    exact = f"{statement} [{chosen['source']}, page {chosen['page']}]."
                    if statement and not verify_answer(exact, [chosen])["issues"]:
                        citation = (chosen["source"], chosen["page"])
                        source_pages = [chosen, *(item for item in evidence if
                                         (item.get("source"), item.get("page")) != citation)]
                        return {"answer": exact, "status": "machine_checked", "issues": [],
                                "claim_checks": [{"id": 1, "text": statement, "original": exact,
                                                  "citations": [citation], "quote": source_statement,
                                                  "verdict": "source_text", "relevant": True, "passed": True}],
                                "candidate_claim_checks": checked["claims"], "chunks": source_pages}
            if plan.exposure:
                summary = compose_sourced_exposure_summary(effective_question, evidence)
                if summary and not verify_answer(summary, evidence)["issues"]:
                    summary_check = check_claims(summary, evidence, request.ollama_model, effective_question)
                    if summary_check["passed"] and answer_covers_plan(plan, summary_check["claims"], evidence):
                        return {"answer": summary, "status": "machine_checked", "issues": [],
                                "claim_checks": summary_check["claims"], "candidate_claim_checks": checked["claims"],
                                "chunks": evidence}
                extracted = extractive_exposure_answer(effective_question, evidence)
                if extracted and not verify_answer(extracted, evidence)["issues"]:
                    return {"answer": extracted, "status": "source_excerpts", "issues": [],
                            "claim_checks": [], "candidate_claim_checks": checked["claims"], "chunks": evidence}
            supported = [claim for claim in checked["claims"] if claim.get("passed")]
            rejected = [claim for claim in checked["claims"] if claim.get("relevant", True) and not claim.get("passed") and claim.get("text")]
            if rejected or checked.get("addresses_question") is False:
                # Repair is exceptional: a normal answer still uses one draft
                # and one check. Search cached lexical records only on failure.
                repair_evidence = list(evidence)
                if hasattr(self.store, "lexical_search"):
                    query = (effective_question + " " + checked.get("missing_aspects", "") + " " +
                             " ".join(item["text"] for item in rejected)[:400])
                    seen = {(item["source"], item["page"]) for item in repair_evidence}
                    additions = []
                    for item in self.store.lexical_search(query, k=8):
                        if focused_sources and item.get("source") not in focused_sources:
                            continue
                        key = (item["source"], item["page"])
                        if key not in seen and len(additions) < 2:
                            additions.append(item)
                            seen.add(key)
                    if hasattr(self.store, "expand_evidence"):
                        additions = self.store.expand_evidence(additions, question=query)
                    repair_evidence.extend(additions)
                try:
                    revised = repair_answer(
                        effective_question, repair_evidence, rejected, checked["issues"], supported,
                        request.ollama_model,
                    )
                    if revised != "I cannot find this in the source documents." and not verify_answer(revised, repair_evidence)["issues"]:
                        combined = "\n\n".join([*(claim["original"] for claim in supported), revised])
                        rechecked = check_claims(combined, repair_evidence, request.ollama_model, effective_question)
                        existing = {claim.get("original", "").casefold() for claim in supported}
                        new_supported = [
                            claim for claim in rechecked["claims"]
                            if claim.get("passed") and claim.get("original", "").casefold() not in existing
                        ]
                        supported.extend(new_supported)
                        if new_supported:
                            evidence = repair_evidence
                            if rechecked["passed"] and answer_covers_plan(plan, supported, evidence):
                                repaired_answer = "\n\n".join(claim["original"] for claim in supported)
                                if not verify_answer(repaired_answer, evidence)["issues"]:
                                    return {"answer": repaired_answer, "status": "machine_checked", "issues": [],
                                            "claim_checks": supported, "candidate_claim_checks": checked["claims"],
                                            "chunks": evidence}
                except Exception:
                    pass
            if supported:
                partial_answer = "\n\n".join(claim["original"] for claim in supported)
                if not verify_answer(partial_answer, evidence)["issues"]:
                    if answer_covers_plan(plan, supported, evidence):
                        return {"answer": partial_answer, "status": "machine_checked", "issues": [],
                                "claim_checks": supported, "candidate_claim_checks": checked["claims"], "chunks": evidence}
                    if not plan.broad:
                        return {
                            "answer": partial_answer, "status": "partially_checked",
                            "issues": checked["issues"], "claim_checks": supported,
                            "candidate_claim_checks": checked["claims"], "chunks": evidence,
                        }
                    # A few individually supported details can still be a
                    # misleading overview when the definition or main aspects
                    # are missing. Do not present them as an answer.
            return {
                "answer": "I could not confirm this answer from the documents. Please check the source pages below or ask a more specific question.",
                "status": "review_required", "issues": checked["issues"], "claim_checks": checked["claims"],
                "candidate_claim_checks": checked["claims"], "chunks": evidence,
            }
        accepted = [claim for claim in checked["claims"] if claim.get("passed")]
        if accepted and all(claim.get("original") for claim in accepted):
            answer = "\n\n".join(claim["original"] for claim in accepted)
        return {"answer": answer, "status": "machine_checked", "issues": [], "claim_checks": accepted,
                "candidate_claim_checks": checked["claims"], "chunks": evidence}


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="PDF directory RAG (Local)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing source PDFs")
    parser.add_argument("--index", action="store_true", help="Rebuild the PDF index")
    parser.add_argument("--ask", help="Ask a question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--topk", type=int, default=6)
    args = parser.parse_args()
    rag = SimpleAssistantRAG(data_dir=args.data_dir, ollama_model=args.model)
    if args.index:
        rag.index()
        print(f"Indexed PDFs in {args.data_dir}")
    if args.ask:
        out = rag.ask(SimpleAssistantRequest(args.ask, args.topk, args.model))
        print("\n=== Answer ===\n")
        print(out["answer"])
        if out.get("issues"):
            print("\nVerifier issues:", out["issues"])
        print("\n=== Evidence ===")
        for item in out["chunks"]:
            print(f"{item['source']}, page {item['page']}: {item['text'][:220]}...")


if __name__ == "__main__":
    main()
