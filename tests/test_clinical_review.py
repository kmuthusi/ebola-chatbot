import csv
import json

from scripts import clinical_review


def test_review_gate_requires_two_current_clinicians(tmp_path, monkeypatch):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"first source revision")
    case = {
        "id": "TEST-1", "domain": "test", "question": "Question?", "expected_behavior": "answer",
        "expected_concepts": [], "prohibited_phrases": [],
        "anchor": {"source": "a.pdf", "page": 1, "quote": "source"},
    }
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([case]), encoding="utf-8")
    monkeypatch.setattr(clinical_review, "CASES_PATH", cases_path)
    reviews = tmp_path / "reviews.csv"
    clinical_review.prepare_reviews([case], tmp_path, reviews)
    assert clinical_review.check_reviews([case], tmp_path, reviews)

    with reviews.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for number, row in enumerate(rows, 1):
        row.update({
            "reviewer_id": f"clinician-{number}", "reviewer_role": "clinician",
            "source_checked": "yes", "expected_answer_correct": "yes",
            "prohibited_claims_correct": "yes", "anchor_correct": "yes",
            "decision": "approve", "reviewed_at": "2026-10-30T12:00:00+03:00",
        })
    with reviews.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=clinical_review.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert clinical_review.check_reviews([case], tmp_path, reviews) == []

    source.write_bytes(b"revised source")
    assert clinical_review.check_reviews([case], tmp_path, reviews)
