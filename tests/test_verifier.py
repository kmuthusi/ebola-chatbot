from verifier import verify_answer


def test_citation_must_match_document_and_page():
    evidence = [{"source": "a.pdf", "page": 4, "text": "A fact."}, {"source": "b.pdf", "page": 4, "text": "Another fact."}]
    assert verify_answer("Fact [a.pdf, page 4].", evidence)["issues"] == []
    assert verify_answer("Fact [c.pdf, page 4].", evidence)["issues"]
    assert verify_answer("Fact [a.pdf, page 5].", evidence)["issues"]
    assert verify_answer("Fact.", evidence)["issues"]
    assert verify_answer("I cannot find this in the source documents.", evidence)["issues"] == []
    assert verify_answer("A fact. I cannot find this in the source documents.", evidence)["issues"]
