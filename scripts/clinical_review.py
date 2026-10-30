"""Validate, screen, and prepare clinician review of the filovirus question set."""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claim_checker import _normalize
from retrievers import PDFDirectoryRetriever
from simple_rag import DEFAULT_MODEL, SimpleAssistantRAG, SimpleAssistantRequest

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "clinical_review" / "cases.json"
REVIEW_PATH = ROOT / "clinical_review" / "reviewer_template.csv"
FIELDS = (
    "case_id", "domain", "question", "expected_behavior", "expected_concepts", "prohibited_phrases",
    "anchor_source", "anchor_page", "anchor_quote",
    "reviewer_id", "reviewer_role", "source_checked", "expected_answer_correct",
    "prohibited_claims_correct", "anchor_correct", "decision", "reviewed_at", "notes",
    "case_set_sha256", "source_sha256",
)


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("The question set must contain unique case IDs.")
    for case in cases:
        if not case.get("question") or case.get("expected_behavior") not in {"answer", "decline"}:
            raise ValueError(f"Invalid case: {case.get('id')}")
    return cases


def source_hash(case: dict, data_dir: Path, corpus_hash: str) -> str:
    anchor = case.get("anchor")
    if not anchor:
        return corpus_hash
    path = data_dir / anchor["source"]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_anchors(cases: list[dict], data_dir: Path) -> None:
    readers = {}
    for case in cases:
        anchor = case.get("anchor")
        if not anchor:
            continue
        path = data_dir / anchor["source"]
        if not path.is_file():
            raise FileNotFoundError(f"{case['id']}: missing {path}")
        if path not in readers:
            readers[path] = PdfReader(path)
        reader = readers[path]
        page = anchor["page"]
        if not isinstance(page, int) or page < 1 or page > len(reader.pages):
            raise ValueError(f"{case['id']}: invalid PDF page {page}")
        text = reader.pages[page - 1].extract_text() or ""
        if _normalize(anchor["quote"]) not in _normalize(text):
            raise ValueError(f"{case['id']}: anchor quote is absent from {path.name}, PDF page {page}")


def prepare_reviews(cases: list[dict], data_dir: Path, output: Path, force: bool = False) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"Review file already exists: {output}. Use --force only to replace an unused template.")
    corpus_hash = PDFDirectoryRetriever(str(data_dir)).fingerprint()
    case_hash = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            for _ in range(2):
                writer.writerow({
                    "case_id": case["id"],
                    "domain": case["domain"],
                    "question": case["question"],
                    "expected_behavior": case["expected_behavior"],
                    "expected_concepts": json.dumps(case["expected_concepts"], ensure_ascii=False),
                    "prohibited_phrases": json.dumps(case["prohibited_phrases"], ensure_ascii=False),
                    "anchor_source": (case.get("anchor") or {}).get("source", ""),
                    "anchor_page": (case.get("anchor") or {}).get("page", ""),
                    "anchor_quote": (case.get("anchor") or {}).get("quote", ""),
                    "case_set_sha256": case_hash,
                    "source_sha256": source_hash(case, data_dir, corpus_hash),
                })


def check_reviews(cases: list[dict], data_dir: Path, reviews_path: Path) -> list[str]:
    corpus_hash = PDFDirectoryRetriever(str(data_dir)).fingerprint()
    case_hash = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    with reviews_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    known_ids = {case["id"] for case in cases}
    issues = []
    for row in rows:
        if row.get("case_id") not in known_ids:
            issues.append(f"Unknown case ID: {row.get('case_id')}")
    for case in cases:
        approvals = set()
        for row in (item for item in rows if item.get("case_id") == case["id"]):
            reviewer = row.get("reviewer_id", "").strip()
            if row.get("case_set_sha256") != case_hash or row.get("source_sha256") != source_hash(case, data_dir, corpus_hash):
                issues.append(f"{case['id']}: review does not match current cases and source PDF")
                continue
            decision = row.get("decision", "").strip().lower()
            if decision == "reject":
                issues.append(f"{case['id']}: a clinician rejected this case; adjudicate and revise it")
                continue
            if decision != "approve":
                continue
            checks = ("source_checked", "expected_answer_correct", "prohibited_claims_correct", "anchor_correct")
            if not reviewer or row.get("reviewer_role", "").strip().lower() != "clinician" or any(row.get(field, "").strip().lower() != "yes" for field in checks):
                issues.append(f"{case['id']}: approval lacks clinician identity or required checks")
                continue
            try:
                datetime.fromisoformat(row.get("reviewed_at", ""))
            except ValueError:
                issues.append(f"{case['id']}: approval lacks a valid ISO review date")
                continue
            approvals.add(reviewer.casefold())
        if len(approvals) < 2:
            issues.append(f"{case['id']}: needs two distinct clinician approvals")
    return issues


def evaluate(cases: list[dict], data_dir: Path, model: str, limit: int | None, output: Path) -> list[dict]:
    rag = SimpleAssistantRAG(data_dir=str(data_dir), ollama_model=model)
    results = []
    for case in cases[:limit]:
        started = time.perf_counter()
        try:
            response = rag.ask(SimpleAssistantRequest(case["question"], ollama_model=model))
        except Exception as exc:
            response = {"answer": "", "status": "service_issue", "issues": [f"{type(exc).__name__}: {exc}"], "chunks": [], "claim_checks": []}
        answer = response["answer"]
        missing = [group for group in case["expected_concepts"] if not any(term.casefold() in answer.casefold() for term in group)]
        prohibited = [term for term in case["prohibited_phrases"] if term.casefold() in answer.casefold()]
        claims = response.get("candidate_claim_checks", response.get("claim_checks", []))
        unsupported_count = sum(claim.get("relevant", True) and not claim.get("passed", False) for claim in claims)
        anchor = case.get("anchor")
        source_present = not anchor or any(chunk.get("source") == anchor["source"] for chunk in response.get("chunks", []))
        behavior_ok = (
            response.get("status") == "machine_checked" if case["expected_behavior"] == "answer"
            else response.get("status") in {"insufficient_context", "review_required"}
        )
        passed = behavior_ok and not missing and not prohibited and source_present
        results.append({
            "case_id": case["id"], "domain": case["domain"], "question": case["question"],
            "passed_automatic_screen": passed, "status": response.get("status"), "answer": answer,
            "missing_concepts": missing, "prohibited_phrases_found": prohibited,
            "answer_complete": not missing and response.get("status") == "machine_checked" and case["expected_behavior"] == "answer",
            "unsupported_claim_count": unsupported_count,
            "all_shown_claims_checked": response.get("status") in {"machine_checked", "partially_checked"},
            "anchor_source_retrieved": source_present, "issues": response.get("issues", []),
            "claim_checks": response.get("claim_checks", []),
            "candidate_claim_checks": response.get("candidate_claim_checks", []),
            "cited_sources": [(item.get("source"), item.get("page")) for item in response.get("chunks", [])],
            "seconds": round(time.perf_counter() - started, 2),
            "clinical_review": "pending",
        })
        print(f"{case['id']}: {'PASS' if passed else 'REVIEW'} ({results[-1]['seconds']}s, {response.get('status')})")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "prepare", "gate", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reviews", type=Path, default=REVIEW_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results" / "clinical_screen.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", help="Evaluate only these case IDs; repeat for multiple cases")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cases = load_cases()
    validate_anchors(cases, args.data_dir)
    if args.action == "validate":
        print(f"Validated {len(cases)} draft cases and their source anchors.")
    elif args.action == "prepare":
        prepare_reviews(cases, args.data_dir, args.reviews, args.force)
        print(f"Wrote two review rows per case to {args.reviews}")
    elif args.action == "gate":
        issues = check_reviews(cases, args.data_dir, args.reviews)
        if issues:
            print("Clinical review pending:\n" + "\n".join(issues))
            raise SystemExit(1)
        print("All cases have two current clinician approvals.")
    else:
        selected_cases = [case for case in cases if not args.case_id or case["id"] in args.case_id]
        if args.case_id and len(selected_cases) != len(set(args.case_id)):
            raise SystemExit("Unknown or duplicate case ID in --case-id")
        selected_cases = selected_cases[:args.limit]
        results = evaluate(selected_cases, args.data_dir, args.model, None, args.output)
        expected_answers = [item for item, case in zip(results, selected_cases) if case["expected_behavior"] == "answer"]
        print(f"Automatic screen: {sum(item['passed_automatic_screen'] for item in results)}/{len(results)} passed. "
              f"Concept coverage: {sum(item['answer_complete'] for item in expected_answers)}/{len(expected_answers)}. "
              f"Unsupported candidate claims: {sum(item['unsupported_claim_count'] for item in results)}. "
              "Clinical review remains pending.")


if __name__ == "__main__":
    main()
