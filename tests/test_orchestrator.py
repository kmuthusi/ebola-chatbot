import simple_rag
from simple_rag import SimpleAssistantRAG, SimpleAssistantRequest, question_with_context


class FakeStore:
    def __init__(self, **kwargs):
        self.indexed = None
        self.fingerprint = None

    def ensure_index(self, chunks):
        self.indexed = chunks

    def source_is_current(self, fingerprint):
        return self.fingerprint == fingerprint

    def mark_source(self, fingerprint):
        self.fingerprint = fingerprint

    def index_chunks(self, chunks):
        self.indexed = chunks

    def query(self, question, k=5, source=None):
        return [{"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis.", "score": 0.9}]

    def search(self, question, k=30):
        return self.query(question, k=k)


def test_follow_up_uses_previous_question_as_topic():
    assert question_with_context("How is it spread?", "Tell me about Ebola") == "How is Ebola spread?"
    assert question_with_context("How about health workers?", "What is guidance on contact tracing?") == (
        "How about health workers in relation to contact tracing?"
    )
    assert question_with_context("What are Marburg symptoms?", "Tell me about Ebola") == "What are Marburg symptoms?"


def test_orchestrator_uses_all_pdf_sources(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis."},
        {"source": "care.pdf", "page": 2, "text": "Supportive care is used."},
    ])
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag, "compose_answer", lambda question, evidence, ollama_model: "PCR is used [testing.pdf, page 3].")
    monkeypatch.setattr(simple_rag, "check_claims", lambda *args: {"passed": True, "issues": [], "claims": [{"id": 1, "verdict": "supported"}]})
    rag = SimpleAssistantRAG(data_dir="data", ollama_model="test")
    result = rag.ask(SimpleAssistantRequest("How is it diagnosed?"))
    assert len(rag.store.indexed) == 2
    assert result["issues"] == []
    assert result["status"] == "machine_checked"
    assert result["chunks"][0]["source"] == "testing.pdf"
    rag.sources.read_and_chunk = lambda: (_ for _ in ()).throw(AssertionError("Unchanged PDFs were extracted again"))
    rag.ask(SimpleAssistantRequest("How is it diagnosed?"))


def test_orchestrator_withholds_unchecked_draft(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "Unsupported medical claim [testing.pdf, page 3].")
    monkeypatch.setattr(simple_rag, "check_claims", lambda *args: {
        "passed": False, "issues": ["Claim 1 unsupported."], "claims": []
    })
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("How is it diagnosed?"))
    assert result["status"] == "review_required"
    assert "Unsupported medical claim" not in result["answer"]
    assert result["chunks"]


def test_orchestrator_withholds_uncited_answer(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "PCR is used for diagnosis.")
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("How is it diagnosed?"))
    assert result["status"] == "review_required"
    assert "No document and page citations" in result["issues"][0]


def test_orchestrator_uses_same_retrieval_path_for_different_questions(monkeypatch):
    seen = []

    class RecordingStore(FakeStore):
        def search(self, question, k=30):
            seen.append(question)
            return super().search(question, k)

    monkeypatch.setattr(simple_rag, "VectorStore", RecordingStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "A fact [testing.pdf, page 3].")
    monkeypatch.setattr(simple_rag, "check_claims", lambda *args: {"passed": True, "issues": [], "claims": []})
    rag = SimpleAssistantRAG()
    for question in ("How is Ebola diagnosed?", "What are the symptoms?", "How does it spread?"):
        assert rag.ask(SimpleAssistantRequest(question))["status"] == "machine_checked"
    assert seen[0] == "How is Ebola diagnosed?"
    assert "What are the symptoms?" in seen
    assert "How does it spread?" in seen
    assert any("diagnostic testing" in query for query in seen)
    assert any("clinical presentation" in query for query in seen)


def test_orchestrator_shows_only_checked_claims(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "A supported fact."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: (
        "Supported fact [testing.pdf, page 3].\n\nUnsupported claim [testing.pdf, page 3]."
    ))
    monkeypatch.setattr(simple_rag, "check_claims", lambda *args: {
        "passed": False, "issues": ["Claim 2 unsupported."], "claims": [
            {"id": 1, "original": "Supported fact [testing.pdf, page 3].", "passed": True},
            {"id": 2, "original": "Unsupported claim [testing.pdf, page 3].", "passed": False},
        ],
    })
    monkeypatch.setattr(simple_rag, "repair_answer", lambda *args: "I cannot find this in the source documents.")
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("A question"))
    assert result["status"] == "machine_checked"
    assert result["answer"] == "Supported fact [testing.pdf, page 3]."


def test_classification_falls_back_to_exact_indexed_definition(monkeypatch):
    class DefinitionStore(FakeStore):
        def search(self, question, k=30):
            return [{"source": "risk-assessment.pdf", "page": 4,
                     "text": "Example disease is a severe form of illness.", "score": .9}]

        def definition_search(self, subject, k=8):
            assert subject == "Example disease"
            return [{"source": "risk-assessment.pdf", "page": 4,
                     "text": "Example disease is a severe form of illness.", "score": .9}]

    monkeypatch.setattr(simple_rag, "VectorStore", DefinitionStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "risk-assessment.pdf", "page": 4, "text": "Example disease is a severe form of illness."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "Example disease is a pathogen [risk-assessment.pdf, page 4].")
    monkeypatch.setattr(simple_rag, "check_claims", lambda *args: {
        "passed": False, "issues": ["Entity mismatch"], "claims": []
    })
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("How does the risk assessment classify Example disease?"))
    assert result["status"] == "machine_checked"
    assert result["answer"] == "Example disease is a severe form of illness [risk-assessment.pdf, page 4]."
    assert result["claim_checks"][0]["verdict"] == "source_text"


def test_orchestrator_repairs_rejected_claim_once(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "PCR is used for diagnosis."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "A vague claim [testing.pdf, page 3].")
    monkeypatch.setattr(simple_rag, "repair_answer", lambda *args: "PCR is used for diagnosis [testing.pdf, page 3].")
    calls = []

    def check(answer, *args):
        calls.append(answer)
        if len(calls) == 1:
            return {"passed": False, "issues": ["Claim 1 unsupported."], "claims": [
                {"id": 1, "text": "A vague claim", "original": answer, "passed": False},
            ]}
        return {"passed": True, "issues": [], "claims": [
            {"id": 1, "text": "PCR is used for diagnosis", "original": answer, "passed": True},
        ]}

    monkeypatch.setattr(simple_rag, "check_claims", check)
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("How is it diagnosed?"))
    assert result["status"] == "machine_checked"
    assert result["answer"] == "PCR is used for diagnosis [testing.pdf, page 3]."
    assert len(calls) == 2


def test_orchestrator_repairs_missing_answer_aspect(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "fingerprint", lambda self: "source-hash")
    monkeypatch.setattr(simple_rag.PDFDirectoryRetriever, "read_and_chunk", lambda self: [
        {"source": "testing.pdf", "page": 3, "text": "The disease is a form of Ebola disease."}
    ])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args: "The disease is severe [testing.pdf, page 3].")
    monkeypatch.setattr(simple_rag, "repair_answer", lambda *args: "It is a form of Ebola disease [testing.pdf, page 3].")
    calls = []

    def check(answer, *args):
        calls.append(answer)
        if len(calls) == 1:
            return {"passed": False, "addresses_question": False, "missing_aspects": "classification",
                    "issues": ["Question not answered"], "claims": [
                        {"id": 1, "text": "The disease is severe", "original": answer, "passed": True},
                    ]}
        return {"passed": True, "addresses_question": True, "issues": [], "claims": [
            {"id": 1, "text": "The disease is severe", "original": calls[0], "passed": True},
            {"id": 2, "text": "It is a form of Ebola disease", "original": "It is a form of Ebola disease [testing.pdf, page 3].", "passed": True},
        ]}

    monkeypatch.setattr(simple_rag, "check_claims", check)
    result = SimpleAssistantRAG().ask(SimpleAssistantRequest("How is it classified?"))
    assert result["status"] == "machine_checked"
    assert "form of Ebola disease" in result["answer"]
    assert len(calls) == 2
