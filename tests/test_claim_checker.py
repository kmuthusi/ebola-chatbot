import json

import claim_checker
from claim_checker import check_claims, split_claims


EVIDENCE = [{"source": "guide.pdf", "page": 2, "text": "The recommended specimen for living patients is whole blood."}]


def test_split_claims_requires_each_sentence_to_stand_alone():
    answer = "Whole blood is used [guide.pdf, page 2]. Another test is used."
    assert len(split_claims(answer)) == 2
    result = check_claims(answer, EVIDENCE, "test")
    assert not result["passed"]
    assert "Claim 2 has no" in result["issues"][0]


def test_inline_bullets_and_numbers_are_checked_separately():
    bullets = "Care includes: • Give fluids [guide.pdf, page 2]. • Give medicine."
    assert len(split_claims(bullets)) == 2
    result = check_claims(bullets, EVIDENCE, "test")
    assert not result["passed"]
    assert "Claim 2 has no document and page citation." in result["issues"]

    numbers = "Steps: 1. Give fluids [guide.pdf, page 2]. 2. Give medicine."
    assert len(split_claims(numbers)) == 2
    result = check_claims(numbers, EVIDENCE, "test")
    assert "Claim 2 has no document and page citation." in result["issues"]


def test_supported_claim_needs_exact_quote(monkeypatch):
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported", "quote": "recommended specimen for living patients is whole blood", "reason": ""
        }]})}
    })
    result = check_claims("Whole blood is recommended for living patients [guide.pdf, page 2].", EVIDENCE, "test")
    assert result["passed"]
    assert result["claims"][0]["verdict"] == "supported"
    assert result["claims"][0]["passed"]


def test_question_entity_must_appear_in_cited_excerpt(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "There are no licensed treatments for Bundibugyo virus disease."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported",
            "quote": "There are no licensed treatments for Bundibugyo virus disease.", "reason": "",
        }]})}
    })
    result = check_claims("There are no licensed treatments for Marburg [guide.pdf, page 2].", evidence, "test", "Is there a treatment for Marburg and Bundibugyo?")
    assert not result["passed"]
    assert "Marburg" in result["issues"][0]
    assert not result["claims"][0]["passed"]


def test_single_named_subject_does_not_require_literal_word_in_excerpt(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "The recommended specimen for orthoebolaviruses is whole blood."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported",
            "quote": "The recommended specimen for orthoebolaviruses is whole blood.", "reason": "",
        }]})}
    })
    result = check_claims("Whole blood is recommended for Ebola [guide.pdf, page 2].", evidence, "test", "What specimen is recommended for Ebola?")
    assert result["passed"]


def test_negative_claim_needs_explicit_exclusion_in_source_quote(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "Ebola spreads through contact with infected body fluids."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported",
            "quote": "Ebola spreads through contact with infected body fluids.", "reason": "",
        }]})}
    })
    result = check_claims("Ebola cannot spread through handshakes [guide.pdf, page 2].", evidence, "test", "Can Ebola spread through handshakes?")
    assert not result["passed"]
    assert "explicit supporting exclusion" in result["issues"][0]


def test_claim_examples_must_appear_in_supporting_quote(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "Contaminated materials include bedding and clothing."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported",
            "quote": "Contaminated materials include bedding and clothing.", "reason": "",
        }]})}
    })
    result = check_claims("Contaminated materials include items such as bedding, clothing, and door knobs [guide.pdf, page 2].", evidence, "test")
    assert not result["passed"]
    assert "door knobs" in result["issues"][0]
    assert claim_checker._unquoted_examples(
        "Ebola is not spread through casual contact like shaking hands.",
        "The virus spreads through contact with infected body fluids.",
    ) == ["shaking hands"]
    assert claim_checker._unquoted_examples(
        "Testing of asymptomatic people, including asymptomatic contacts, is not recommended.",
        "Testing of asymptomatic people, including asymptomatic contacts, is not recommended.",
    ) == []
    assert claim_checker._unquoted_examples(
        "Balanced crystalloids such as Ringer's lactate for resuscitation are recommended.",
        "Balanced crystalloids such as Ringer's lactate are recommended.",
    ) == []


def test_supported_facts_can_still_miss_question(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "The disease is severe. It is a form of Ebola disease."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({
            "claims": [{"id": 1, "verdict": "supported", "quote": "The disease is severe.", "reason": ""}],
            "addresses_question": False, "missing_aspects": "classification as Ebola disease",
        })}
    })
    result = check_claims("The disease is severe [guide.pdf, page 2].", evidence, "test", "How is it classified?")
    assert not result["passed"]
    assert not result["addresses_question"]
    assert result["claims"][0]["passed"]


def test_unrelated_background_is_not_returned_as_answer(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "Advice should fit the local context. The fatality rate is 30%."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({
            "claims": [
                {"id": 1, "verdict": "supported", "quote": "Advice should fit the local context.", "relevant": True},
                {"id": 2, "verdict": "supported", "quote": "The fatality rate is 30%.", "relevant": False},
            ],
            "addresses_question": True,
        })}
    })
    result = check_claims(
        "Advice should fit the local context [guide.pdf, page 2].\n\nThe fatality rate is 30% [guide.pdf, page 2].",
        evidence, "test", "How should advice be adapted?",
    )
    assert result["passed"]
    assert [claim["passed"] for claim in result["claims"]] == [True, False]
    assert claim_checker._is_unrequested_statistic(
        "How should advice be adapted?", "The case fatality rate is 30%."
    )
    assert not claim_checker._is_unrequested_statistic(
        "What is the case fatality rate?", "The case fatality rate is 30%."
    )
    assert not claim_checker._is_unrequested_statistic(
        "Which fluid is recommended?", "Balanced crystalloid is preferred over 0.9% saline."
    )
    assert claim_checker._is_unrequested_definition(
        "How should advice be adapted?", "Bundibugyo disease is a severe form of Ebola disease."
    )
    assert not claim_checker._is_unrequested_definition(
        "How is Bundibugyo disease classified?", "Bundibugyo disease is a form of Ebola disease."
    )
    assert claim_checker._strong_topic_match(
        "Which intravenous fluid is recommended for acute resuscitation?",
        "Balanced crystalloids are recommended for acute intravenous fluid resuscitation.",
    )


def test_unsafe_activity_label_does_not_prove_transmission(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "Greetings by touching, such as shaking hands, were called unsafe practices."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported", "quote": "Greetings by touching, such as shaking hands, were called unsafe practices.",
            "relevant": True,
        }]})}
    })
    result = check_claims("You can get Ebola by shaking hands [guide.pdf, page 2].", evidence, "test", "Can I get Ebola by shaking hands?")
    assert not result["passed"]
    assert "explicit route evidence" in result["issues"][0]
    assert not claim_checker._exposure_route_supported(
        "You can get Ebola by shaking hands with someone infected.",
        "Ebola spreads through direct contact with infected body fluids.",
    )
    assert not claim_checker._specific_exposure_supported(
        "Can I get Ebola by shaking hands?",
        "Shaking hands can be a way to get infected through contact.",
        "Indirect contact involves a contaminated intermediate object, such as a contaminated hand, that transfers microorganisms.",
    )
    assert claim_checker._specific_exposure_supported(
        "Can I get a refund by applying online?",
        "You can get a refund by applying online.",
        "Applicants are eligible to receive a refund by applying online.",
    )


def test_one_supported_item_cannot_validate_whole_equipment_list(monkeypatch):
    evidence = [{"source": "guide.pdf", "page": 2, "text": "Their footwear should be waterproof boots."}]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported", "quote": "Their footwear should be waterproof boots.",
        }]})}
    })
    result = check_claims(
        "Cleaners should wear a mask, two pairs of gloves, and waterproof boots [guide.pdf, page 2].",
        evidence, "test", "What should cleaners wear?",
    )
    assert not result["passed"]
    assert "lists items absent" in result["issues"][0]


def test_hallucinated_quote_and_unsupported_verdict_are_held(monkeypatch):
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "supported", "quote": "a fabricated quotation that is absent", "reason": ""
        }]})}
    })
    assert not check_claims("A claim [guide.pdf, page 2].", EVIDENCE, "test")["passed"]

    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {
        "message": {"content": json.dumps({"claims": [{
            "id": 1, "verdict": "unsupported", "quote": "", "reason": "not stated"
        }]})}
    })
    assert not check_claims("A claim [guide.pdf, page 2].", EVIDENCE, "test")["passed"]


def test_quote_matching_tolerates_pdf_line_breaks_and_one_typo():
    source = "The recommended specimen is whole blood for living patients and oral swab for deceased individuals."
    quote = "The recommended specimen is whole bl0od for living patients and oral swab for deceased individuals"
    assert claim_checker._quote_matches(quote, [source])
    assert not claim_checker._quote_matches("An unrelated fabricated recommendation about treatment", [source])
    assert claim_checker._quote_matches(
        "The recommended specimen is whole blood ... oral swab for deceased individuals",
        [source],
    )

def test_malformed_or_incomplete_model_review_fails_closed(monkeypatch):
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {"message": {"content": "not JSON"}})
    result = check_claims("Whole blood is used [guide.pdf, page 2].", EVIDENCE, "test")
    assert not result["passed"]
    monkeypatch.setattr(claim_checker.ollama, "chat", lambda **kwargs: {"message": {"content": '{"claims":[]}'}})
    assert not check_claims("Whole blood is used [guide.pdf, page 2].", EVIDENCE, "test")["passed"]
