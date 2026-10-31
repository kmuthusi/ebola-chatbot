from query_planning import make_search_plan, corpus_facets, focus_on_matching_sources, answer_covers_plan, focus_excerpt, classification_target, comparison_subjects
from rag_pipeline import has_source_anchor, is_feedback_statement, rerank_and_release
from retrievers import _is_reference_page
from evidence_selector import extractive_answer


def test_broad_plan_covers_distinct_facets_without_topic_answers():
    plan = make_search_plan("Tell me about contact tracing")
    assert plan.broad
    assert all("contact tracing" in query for query in plan.queries)
    assert len(plan.queries) > 3


def test_focused_plan_expands_the_information_need():
    plan = make_search_plan("How is Ebola diagnosed?")
    assert not plan.broad
    assert "diagnostic testing" in plan.queries[1]
    assert "specimen sample" in plan.queries[2]
    assert not make_search_plan("What are the symptoms?").broad


def test_comparison_plan_searches_both_subjects_for_any_topic():
    for question in ("How does Example A differ from Example B?",
                     "What is the difference between Example A and Example B?",
                     "Compare Example A and Example B"):
        assert comparison_subjects(question) == ("Example A", "Example B")
        plan = make_search_plan(question)
        assert plan.comparison == ("Example A", "Example B")
        assert any("Example A definition" in query for query in plan.queries)
        assert any("Example B definition" in query for query in plan.queries)


def test_extractive_fallback_copies_retrieved_sentences_without_topic_facts():
    evidence = [
        {"source": "activity.pdf", "page": 2,
         "text": "During an Example outbreak, hugging was described as an unsafe greeting."},
        {"source": "route.pdf", "page": 4,
         "text": "Person-to-person spread occurs through direct contact with droplets."},
    ]
    answer = extractive_answer("Can I get Example by hugging?", evidence, (
        "hugging unsafe greeting", "person-to-person spread direct contact"))
    assert "hugging was described as an unsafe greeting [activity.pdf, page 2]" in answer
    assert "spread occurs through direct contact with droplets [route.pdf, page 4]" in answer
    assert "Ebola" not in answer


def test_facets_follow_indexed_definition_for_any_subject():
    disease = corpus_facets("Example", "Example is a viral disease affecting people.")
    process = corpus_facets("Example", "Example is a process for finding contacts.")
    assert "symptoms include" in disease[0]
    assert "how it works" in process[1]


def test_broad_answer_must_cite_definition_and_available_aspects():
    plan = make_search_plan("Tell me about Example disease")
    evidence = [
        {"source": "guide.pdf", "page": 1, "is_definition": True},
        {"source": "guide.pdf", "page": 2, "facet": "aspect_0"},
        {"source": "guide.pdf", "page": 3, "facet": "aspect_1"},
        {"source": "guide.pdf", "page": 4, "facet": "aspect_2"},
    ]
    claims = [
        {"passed": True, "citations": [("guide.pdf", 1)]},
        {"passed": True, "citations": [("guide.pdf", 5)]},
        {"passed": True, "citations": [("guide.pdf", 6)]},
    ]
    assert not answer_covers_plan(plan, claims, evidence)
    claims[1]["citations"] = [("guide.pdf", 2)]
    claims[2]["citations"] = [("guide.pdf", 3)]
    assert not answer_covers_plan(plan, claims, evidence)
    claims.append({"passed": True, "citations": [("guide.pdf", 4)]})
    assert answer_covers_plan(plan, claims, evidence)


def test_document_title_guides_focused_search_when_enough_matches_exist():
    plan = make_search_plan("Example treatment")
    hits = ([{"source": "clinical-management.pdf", "page": n, "text": "care"} for n in range(6)]
            + [{"source": "diagnostic-testing.pdf", "page": 9, "text": "sample"}])
    assert {item["source"] for item in focus_on_matching_sources(plan, hits)} == {"clinical-management.pdf"}
    assert focus_on_matching_sources(plan, hits[:2] + hits[-1:]) == hits[:2] + hits[-1:]


def test_exposure_plan_pairs_named_activity_with_documented_route():
    plan = make_search_plan("Can I get Ebola by shaking hands?")
    assert plan.exposure
    assert plan.subject == "ebola"
    assert "shaking hands" in plan.queries[1]
    assert "transmission" in plan.queries[2]


def test_classification_target_is_extracted_without_topic_specific_rules():
    assert classification_target("How does this guide classify Example disease?") == "Example disease"
    assert classification_target("How does it spread?") is None


def test_exposure_answer_needs_activity_guidance_and_separate_route():
    plan = make_search_plan("Can I get Ebola by shaking hands?")
    activity = {"text": "Shaking hands is considered an unsafe greeting.", "passed": True}
    route = {"text": "Transmission occurs through contact with body fluids.", "passed": True}
    assert not answer_covers_plan(plan, [activity], [])
    assert answer_covers_plan(plan, [activity, route], [])


def test_focused_excerpt_starts_at_sentence_boundary():
    text = "Context sentence. When people were asked about an unsafe practice, they named shaking hands. More unrelated detail follows."
    assert focus_excerpt(text, "shaking hands unsafe practice", limit=100).startswith("When people")


def test_recommendation_outranks_research_agenda_for_care_query():
    candidates = [
        {"source": "a.pdf", "page": 1, "score": .65, "text": "Ebola treatment is an area of future research."},
        {"source": "b.pdf", "page": 2, "score": .60, "text": "WHO recommends supportive care and monitoring for Ebola treatment."},
    ]
    result = rerank_and_release("Ebola treatment", candidates, "documents", max_passages=2)
    assert result["evidence"][0]["source"] == "b.pdf"


def test_bibliography_page_is_excluded_even_without_references_heading():
    text = "33. Smith AB et al. Study of a disease treatment. doi: 10.1234/example\n34. Jones C et al. Another study. J Infect. 2020."
    assert _is_reference_page(text)


def test_scope_requires_a_substantive_term_in_retrieved_sources():
    hits = [{"source": "care.pdf", "page": 2,
             "text": "Ebola disease can have a high case fatality rate."}]
    assert has_source_anchor("How deadly is Ebola?", hits)
    assert not has_source_anchor("This is embarrassing", hits)
    assert not has_source_anchor("What are Xalec service options?", hits)
    assert is_feedback_statement("This is embarrassing")
    assert not is_feedback_statement("How is this different from Marburg?")
