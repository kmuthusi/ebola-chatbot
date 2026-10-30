from rag_pipeline import rerank_and_release

def test_rerank_and_release_grounded():
    evidence = [{"source": "a.pdf", "page": 1, "text": "Relevant evidence", "score": 0.85}]
    out = rerank_and_release("What does the source say?", evidence, intent="documents")
    assert out["release"] == "grounded"
    assert len(out["evidence"]) >= 1

def test_rerank_and_release_refusal():
    candidates = [{"page": None, "text": "irrelevant", "score": 0.1}]
    out = rerank_and_release("?", candidates, intent="documents")
    assert out["release"] == "refusal"


def test_passage_limit_and_page_diversity():
    candidates = [
        {"source": "a.pdf", "page": 1, "text": "Ebola diagnosis A", "score": 0.9},
        {"source": "a.pdf", "page": 1, "text": "Ebola diagnosis B", "score": 0.89},
        {"source": "b.pdf", "page": 4, "text": "Ebola diagnosis C", "score": 0.8},
    ]
    result = rerank_and_release("Ebola diagnosis", candidates, "documents", max_passages=2)
    assert [(item["source"], item["page"]) for item in result["evidence"]] == [("a.pdf", 1), ("b.pdf", 4)]


def test_exact_question_phrase_can_outrank_generic_word_overlap():
    candidates = [
        {"source": "a.pdf", "page": 1, "text": "Hand hygiene and handwashing advice.", "score": 0.7},
        {"source": "b.pdf", "page": 2, "text": "Shaking hands is described as an unsafe greeting practice.", "score": 0.68},
    ]
    result = rerank_and_release("Can shaking hands spread infection?", candidates, "documents", max_passages=2)
    assert result["evidence"][0]["source"] == "b.pdf"


def test_exposure_question_puts_route_next_to_named_activity():
    candidates = [
        {"source": "activity.pdf", "page": 1, "text": "Shaking hands was called an unsafe greeting practice.", "score": 0.9},
        {"source": "hygiene.pdf", "page": 2, "text": "Hand hygiene stations should be available.", "score": 0.85},
        {"source": "route.pdf", "page": 3, "text": "Infection spreads through direct contact with body fluids.", "score": 0.7},
    ]
    result = rerank_and_release("Can I get infection by shaking hands?", candidates, "documents", max_passages=3)
    assert [item["source"] for item in result["evidence"][:2]] == ["activity.pdf", "route.pdf"]
from rag_pipeline import rerank_and_release


def test_pipeline_reranks_and_releases_relevant_evidence():
    result = rerank_and_release(
        "question",
        [{"page": 2, "text": "evidence", "score": 0.8}, {"page": 3, "text": "other", "score": 0.3}],
        "documents",
    )
    assert result["release"] == "grounded"
    assert result["evidence"][0]["page"] == 2


def test_pipeline_refuses_empty_evidence():
    assert rerank_and_release("question", [], "documents")["release"] == "refusal"


def test_pipeline_prefers_passage_with_distinctive_question_term():
    candidates = [
        {"page": 1, "text": "Ebola and Marburg disease overview", "score": 0.75},
        {"page": 12, "text": "Ebola and Marburg vaccines are specific to each virus", "score": 0.70},
    ]
    result = rerank_and_release("Is there a vaccine for Ebola and Marburg?", candidates, "documents")
    assert result["evidence"][0]["page"] == 12
