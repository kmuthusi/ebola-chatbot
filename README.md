# Ebola and filovirus PDF RAG

Local question answering over **all PDFs in `data/`**. PDF text is embedded with Ollama and stored in ChromaDB. Answers cite both the source filename and page number.

## Setup

Requires Python 3.11+ and a running Ollama service with `nomic-embed-text:latest` and `llama3.1:8b` installed.

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull llama3.1:8b
```

Put source PDFs in `data/`. The legacy Xalec PDF at the repository root is not indexed. Adding, removing, or editing a PDF triggers a rebuild on the next question. Files with no extractable text require OCR before they can be searched.

## Run

```bash
python simple_rag.py --index
python simple_rag.py --ask "How is Ebola virus disease diagnosed?"
streamlit run app.py
```

`python scripts/index_pdfs.py` is an alternative to `python simple_rag.py --index`; either command rebuilds the same index.

The Streamlit app keeps a chat history for the current source directory and model. Its sidebar shows the source PDFs and index status, provides index and chat controls, and lets you choose the answer model and evidence count. Open **See the source pages** beneath an answer to review retrieved passages with filenames and PDF page numbers. Search combines semantic and keyword matches across all indexed PDFs. Broad questions gather a definition and several distinct facets from the indexed text. Focused questions also search related terminology and prefer documents whose filenames match the requested topic. A bounded local passage selector handles large candidate sets. The requested evidence count is a maximum, with distinct pages preferred and nearby text restored within a small prompt budget. The answer model selects numbered source passages; the app attaches their citations. A second model checks each claim and whether the answer addresses the question. Unsupported or unrelated claims are withheld. On a failed claim or incomplete answer, the app makes at most one targeted repair attempt using cached lexical search. For some exposure questions, a source-derived summary is independently checked; exact source excerpts are shown if that check fails. A pass is an automated screen, not clinical approval. No medical answers are stored as templates in the code.

Use `--data-dir` to select another PDF directory, `--topk` for more evidence chunks, and `--model` for another installed chat model. The persisted database is in `chroma_db/` and the current collection is `ebola_sources_ollama`.

## Check

```bash
pytest -q
python scripts/evaluate_queries.py --output evaluation.json
python scripts/clinical_review.py validate
python scripts/clinical_review.py evaluate
python scripts/clinical_review.py gate --reviews clinical_review/reviews.csv
```

The first question after a source change or index-schema change can take longer because it rebuilds the index. Subsequent questions hash PDF bytes to check freshness without extracting every page again. Repeated semantic queries are cached in memory. Reference list pages and publication boilerplate are excluded from ordinary search. The second model pass adds latency and can reject supported answers when extracted PDF text is noisy. The evaluation records concept coverage, unsupported candidate claims, and response seconds separately; use `--case-id FIL-014` to rerun an individual draft case. The [clinical review workflow](clinical_review/README.md) describes the draft, source-anchored question set and the required independent review.
