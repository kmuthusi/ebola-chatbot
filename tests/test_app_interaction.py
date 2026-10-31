from pathlib import Path

from streamlit.testing.v1 import AppTest

import simple_rag


def test_question_is_visible_to_pipeline_before_answer_and_input_recovers(monkeypatch):
    observed = []

    class FakeSources:
        def pdf_paths(self):
            return [Path("guide.pdf")]

        def fingerprint(self):
            return "current"

    class FakeStore:
        def source_is_current(self, fingerprint):
            return True

    class FakeRag:
        def __init__(self, **kwargs):
            self.sources = FakeSources()
            self.store = FakeStore()

        def ask(self, request):
            import streamlit as st

            observed.append((request.question, [message["role"] for message in st.session_state.messages],
                             bool(st.session_state.pending_question)))
            return {"answer": "A source-backed answer [guide.pdf, page 1].", "status": "machine_checked",
                    "chunks": [], "issues": [], "claim_checks": []}

    monkeypatch.setattr(simple_rag, "SimpleAssistantRAG", FakeRag)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    question = next(widget for widget in app.text_input if widget.label == "Your question")
    question.set_value("What does the guide say?")
    ask = next(widget for widget in app.button if widget.label == "Ask")
    app = ask.click().run()
    assert not app.exception
    assert observed == [("What does the guide say?", ["user"], True)]
    assert [message["role"] for message in app.session_state.messages] == ["user", "assistant"]
    assert not app.session_state.pending_question
    question = next(widget for widget in app.text_input if widget.label == "Your question")
    assert not question.disabled
    assert question.value == ""
