import json

import bounded_composer
from bounded_composer import _make_prompt, _from_model_json


EVIDENCE = [{"source": "diagnostic.pdf", "page": 4, "text": "Test evidence."}]


def test_prompt_numbers_source_passages():
    prompt = _make_prompt("What is tested?", EVIDENCE)
    assert "Source 1 (diagnostic.pdf, page 4)" in prompt
    assert "Test evidence." in prompt
    assert "everyday words" in prompt
    assert "Ebola and Marburg" not in prompt
    assert "Answer this question directly: What is tested?" in prompt


def test_composer_builds_citation_from_selected_source(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": json.dumps({"claims": [
            {"text": "The test uses a blood sample", "source_id": 1}
        ]})}}

    monkeypatch.setattr(bounded_composer.ollama, "chat", fake_chat)
    answer = bounded_composer.compose_answer("What is tested?", EVIDENCE, "test-model")
    assert answer == "The test uses a blood sample [diagnostic.pdf, page 4]."
    assert captured["format"]["properties"]["claims"]["maxItems"] == 4


def test_composer_rejects_unretrieved_source_and_uncited_prose():
    for claim in (
        {"text": "An answer", "source_id": 2},
        {"text": "An answer [other.pdf, page 9]", "source_id": 1},
    ):
        try:
            _from_model_json(json.dumps({"claims": [claim]}), EVIDENCE)
        except ValueError:
            pass
        else:
            raise AssertionError("Unbound claim was accepted")


def test_composer_returns_no_answer_when_no_claim_is_supported():
    assert _from_model_json('{"claims":[]}', EVIDENCE) == "I cannot find this in the source documents."


def test_composer_limits_model_output_to_four_sourced_claims():
    claims = [{"text": f"Claim {number}", "source_id": 1} for number in range(5)]
    answer = _from_model_json(json.dumps({"claims": claims}), EVIDENCE)
    assert answer.count("[diagnostic.pdf, page 4]") == 4


def test_specific_question_uses_shorter_claim_budget():
    assert bounded_composer._claim_limit("How should guidance be adapted?") == 2
    assert bounded_composer._claim_limit("How are cases diagnosed and treated?") == 4


def test_composer_drops_duplicate_claims():
    claims = [
        {"text": "A fact", "source_id": 1},
        {"text": "A fact.", "source_id": 1},
    ]
    answer = _from_model_json(json.dumps({"claims": claims}), EVIDENCE)
    assert answer.count("[diagnostic.pdf, page 4]") == 1
