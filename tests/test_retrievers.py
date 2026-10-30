from retrievers import PDFDirectoryRetriever


def test_retriever_reads_each_pdf_and_preserves_source(tmp_path, monkeypatch):
    # Use a small fake reader so this tests discovery and provenance without relying on PDF text extraction internals.
    for name in ("b.PDF", "a.pdf"):
        (tmp_path / name).write_bytes(b"pdf")
    (tmp_path / "ignore.txt").write_text("ignored")
    import retrievers

    class Page:
        def extract_text(self):
            return "A useful source page."

    class Reader:
        def __init__(self, path):
            self.pages = [Page()]

    monkeypatch.setattr(retrievers, "PdfReader", Reader)
    pages = PDFDirectoryRetriever(str(tmp_path)).read_and_chunk()
    assert [(p["source"], p["page"]) for p in pages] == [("a.pdf", 1), ("b.PDF", 1)]


def test_fingerprint_changes_when_pdf_changes(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"first")
    retriever = PDFDirectoryRetriever(str(tmp_path))
    first = retriever.fingerprint()
    path.write_bytes(b"second")
    assert retriever.fingerprint() != first
