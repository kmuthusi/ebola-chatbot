"""Local RAG over all PDFs in a source directory."""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

from retrievers import DEFAULT_DATA_DIR, PDFDirectoryRetriever
from vector_store import VectorStore
from rag_pipeline import rerank_and_release
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
        candidates = self.store.search(effective_question, k=max(request.top_k, 25))
        pipeline = rerank_and_release(
            effective_question, candidates, intent="documents", max_passages=max(1, min(request.top_k, 8)),
        )
        evidence = pipeline["evidence"]
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
        verified = verify_answer(answer, evidence)
        if answer.strip() == "I cannot find this in the source documents.":
            return {"answer": answer, "status": "insufficient_context", "issues": [], "claim_checks": [], "chunks": evidence}
        if verified["issues"]:
            return {
                "answer": "I could not confirm this answer from the documents. Please check the source pages below or ask a more specific question.",
                "status": "review_required", "issues": verified["issues"], "claim_checks": [], "chunks": evidence,
            }
        checked = check_claims(answer, evidence, request.ollama_model, effective_question)
        if not checked["passed"]:
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
                            if rechecked["passed"]:
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
                    return {
                        "answer": partial_answer, "status": "partially_checked",
                        "issues": checked["issues"], "claim_checks": supported,
                        "candidate_claim_checks": checked["claims"], "chunks": evidence,
                    }
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
