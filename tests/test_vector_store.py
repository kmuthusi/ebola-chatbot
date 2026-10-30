from vector_store import VectorStore, _join_overlapping_chunks


def test_page_chunk_overlap_is_removed():
    first = "A recommendation says the outer pair of gloves should be heavy-duty (utility) gloves."
    second = "outer pair of gloves should be heavy-duty (utility) gloves. Their footwear should be boots."
    joined = _join_overlapping_chunks([first, second])
    assert joined.count("outer pair of gloves") == 1
    assert "Their footwear should be boots" in joined

def test_index_and_query(tmp_path):
    store = VectorStore(collection_name="test_coll", chroma_dir=str(tmp_path))
    raw = [{"source": "a.pdf", "page": 1, "text": "Ebola disease diagnostic testing uses laboratory methods."},
           {"source": "b.pdf", "page": 2, "text": "Clinical care includes supportive treatment for patients."}]
    store.index_chunks(raw)
    res = store.query("supportive treatment", k=3)
    assert len(res) >= 1
    assert any(r.get("page") == 2 for r in res)
import vector_store
from vector_store import VectorStore


def test_chunk_text_handles_boundaries(tmp_path):
    store = VectorStore(chroma_dir=str(tmp_path), embed_model="test")
    text = "a" * 320
    chunks = store._chunk_text(text, chunk_len=120, overlap=20)
    assert len(chunks) == 3
    assert all(chunk for chunk in chunks)


def test_index_and_query_use_ollama_batches(tmp_path, monkeypatch):
    calls = []

    def fake_embed(model, input):
        calls.append((model, input))
        return {"embeddings": [[1.0, 0.0, 0.0] for _ in input]}

    monkeypatch.setattr(vector_store.ollama, "embed", fake_embed)
    store = VectorStore(chroma_dir=str(tmp_path), embed_model="test")
    raw = [{"source": "a.pdf", "page": 7, "text": "x" * 220}]
    store.ensure_index(raw)
    assert store.collection.count() == 1
    assert calls[0][0] == "test"
    call_count = len(calls)
    store.ensure_index(raw)
    assert len(calls) == call_count

    result = store.query("question", k=1)
    assert result[0]["page"] == 7
    assert result[0]["source"] == "a.pdf"
    assert result[0]["score"] > 0
    calls_after_query = len(calls)
    store.query("question", k=1)
    assert len(calls) == calls_after_query


def test_query_excludes_reference_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store.ollama, "embed", lambda model, input: {
        "embeddings": [[1.0, 0.0, 0.0] for _ in (input if isinstance(input, list) else [input])]
    })
    store = VectorStore(chroma_dir=str(tmp_path), embed_model="test")
    store.index_chunks([
        {"source": "a.pdf", "page": 1, "section": "body", "text": "Clinical guidance " * 15},
        {"source": "a.pdf", "page": 20, "section": "references", "text": "Citation list " * 15},
    ])
    results = store.query("clinical guidance", k=5, source="a.pdf")
    assert results and {item["page"] for item in results} == {1}


def test_hybrid_search_recovers_a_keyword_match_outside_semantic_results(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store.ollama, "embed", lambda model, input: {
        "embeddings": [[1.0, 0.0, 0.0] for _ in (input if isinstance(input, list) else [input])]
    })
    store = VectorStore(chroma_dir=str(tmp_path), embed_model="test")
    store.index_chunks([
        {"source": "a.pdf", "page": 1, "text": "A general overview of the response process. " * 5},
        {"source": "b.pdf", "page": 9, "text": "Contact tracing finds people who may have been exposed. " * 5},
    ])
    monkeypatch.setattr(store, "query", lambda *args, **kwargs: [{
        "source": "a.pdf", "page": 1, "text": "A general overview of the response process. " * 5,
        "section": "body", "score": 0.7,
    }])
    results = store.search("contact tracing", k=2)
    assert any(item["source"] == "b.pdf" and item["page"] == 9 for item in results)


def test_page_expansion_and_lexical_retry_use_cached_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store.ollama, "embed", lambda model, input: {
        "embeddings": [[1.0, 0.0, 0.0] for _ in (input if isinstance(input, list) else [input])]
    })
    store = VectorStore(chroma_dir=str(tmp_path), embed_model="test")
    text = "Ebola testing uses a specimen. " * 70
    store.index_chunks([{"source": "testing.pdf", "page": 8, "text": text}])
    hits = store.lexical_search("Ebola specimen", k=2)
    assert hits and hits[0]["source"] == "testing.pdf"
    expanded = store.expand_evidence([hits[0]])
    assert expanded[0]["source"] == "testing.pdf"
    assert len(expanded[0]["text"]) >= len(hits[0]["text"])
