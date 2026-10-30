"""Streamlit chat interface for the local filovirus PDF index."""

import re
import unicodedata
from html import escape

import streamlit as st

from answer_formatting import format_answer_markdown
from simple_rag import DEFAULT_MODEL, SimpleAssistantRAG, SimpleAssistantRequest


def equation_to_latex(equation: str) -> str:
    """Convert common Unicode equation notation to display LaTeX."""
    value = unicodedata.normalize("NFKC", " ".join(equation.splitlines()))
    value = re.sub(r"z\s*⊤\s*itbi", r"z^{\top}_{it} b_i", value)
    value = value.replace("εit", r"\epsilon_{it}")
    value = re.sub(r"\b([yxf])it\b", r"\1_{it}", value)
    value = re.sub(r"\b([x])\s+it\b", r"\1_{it}", value)
    return value.replace("𝜙", r"\phi")


def render_document_equation(equation: str) -> None:
    """Render an extracted equation while retaining its original text on failure."""
    try:
        st.latex(equation_to_latex(equation))
    except Exception:
        st.code(equation)


def render_response(result: dict) -> None:
    if result.get("error"):
        st.error(result["answer"])
        return

    st.markdown(format_answer_markdown(result["answer"]))
    if result.get("status") == "partially_checked":
        st.caption("Partial answer: Some details could not be verified and were omitted. Check the cited pages before using this for care.")
        with st.expander("Source check details"):
            for issue in result.get("issues", []):
                st.write(issue)
    elif result.get("issues"):
        st.caption("The source check could not confirm an answer. Check the pages below or try a more specific question.")
        with st.expander("Source check details"):
            for issue in result["issues"]:
                st.write(issue)
    elif result["answer"] == "I cannot find this in the source documents.":
        st.caption("No clear answer was found in the indexed documents. Try a more specific question.")
    elif result.get("status") == "source_excerpts":
        st.caption("These words are copied from the cited pages. They describe the guidance and transmission route; they do not measure the risk from one encounter.")
    elif result.get("status") == "machine_checked":
        st.caption("The answer was checked against the cited pages by software. Check the source text before using it to make a care decision.")
    else:
        st.caption("Check the cited source text before using this answer to make a care decision.")

    if result.get("status") in ("machine_checked", "partially_checked") and result.get("claim_checks"):
        with st.expander("How this answer was checked", expanded=False):
            for claim in result["claim_checks"]:
                st.write(claim.get("text", ""))
                st.caption(f"Matching words in the document: {claim.get('quote', '')}")

    chunks = result.get("chunks", [])
    if chunks:
        with st.expander(f"See the source pages ({len(chunks)})", expanded=False):
            for number, chunk in enumerate(chunks, 1):
                source = chunk.get("source") or "Unknown document"
                page = chunk.get("page") or "?"
                with st.container(border=True):
                    st.write(f"{number}. {source} · page {page}")
                    score = chunk.get("score")
                    if isinstance(score, (int, float)):
                        st.caption(f"Search match: {score:.2f}")
                    st.markdown(
                        f'<div class="source-excerpt">{escape(chunk.get("text", ""))}</div>',
                        unsafe_allow_html=True,
                    )

    equations = []
    seen = set()
    for chunk in chunks:
        for equation in chunk.get("equations", []):
            key = (chunk.get("source"), chunk.get("page"), equation)
            if key not in seen:
                seen.add(key)
                equations.append(key)
    if equations:
        with st.expander("Equations from the documents", expanded=False):
            for source, page, equation in equations:
                render_document_equation(equation)
                st.caption(f"Exact extracted source: {source}, page {page}")


st.set_page_config(page_title="Ebola and Marburg Guidelines Assistant", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ink: #18323b;
        --muted: #536975;
        --line: #dce6e9;
    }
    .stApp { background: #f7f9fa; color: var(--ink); }
    .block-container { max-width: 940px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1 { font-size: clamp(1.9rem, 4vw, 2.5rem) !important; letter-spacing: -.035em; line-height: 1.15; }
    h2, h3 { color: var(--ink); }
    [data-testid="stSidebar"] { background: #f0f5f5; border-right: 1px solid var(--line); }
    [data-testid="stChatMessage"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; }
    .source-excerpt { color: #344b56; font-size: .94rem; line-height: 1.58; white-space: pre-wrap; overflow-wrap: anywhere; }
    button[kind="primaryFormSubmit"] { background: #176c75 !important; border-color: #176c75 !important; color: #fff !important; }
    button[kind="primaryFormSubmit"]:hover { background: #11545c !important; border-color: #11545c !important; }
    @media (max-width: 640px) {
        .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
        [data-testid="stChatMessage"] { padding: .7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Ebola and Marburg Guidelines Assistant")
st.caption("Ask a question in your own words. Answers come from the documents and show where the information was found.")

with st.sidebar:
    st.header("Document library")
    with st.expander("Advanced settings"):
        data_dir = st.text_input("PDF directory", value="data", autocomplete="off")
        model = st.text_input("Ollama answer model", value=DEFAULT_MODEL, autocomplete="off")
        top_k = st.number_input("Source passages to search", min_value=1, max_value=8, value=6, step=1)

    settings_key = (data_dir, model)
    if st.session_state.get("rag_settings") != settings_key:
        st.session_state.rag = SimpleAssistantRAG(data_dir=data_dir, ollama_model=model)
        st.session_state.rag_settings = settings_key
        st.session_state.messages = []
    rag = st.session_state.rag

    try:
        pdfs = rag.sources.pdf_paths()
        index_ready = rag.store.source_is_current(rag.sources.fingerprint())
    except (OSError, ValueError) as exc:
        pdfs = []
        index_ready = False
        st.error(str(exc))

    if index_ready:
        st.success(f"Ready to answer · {len(pdfs)} documents")
    elif pdfs:
        st.info(f"{len(pdfs)} documents found. They will be prepared when you ask a question.")

    if pdfs:
        with st.expander("Documents used for answers"):
            for pdf in pdfs:
                st.write(pdf.name)

    st.divider()
    if st.button("Refresh documents", disabled=not pdfs, use_container_width=True):
        try:
            with st.spinner("Preparing the documents..."):
                rag.index()
            st.success("Documents refreshed.")
        except Exception as exc:
            st.error(f"Could not refresh documents: {exc}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("How to use this assistant"):
        st.write("Ask about Ebola or Marburg disease. Each answer points to the document and page it came from. Open the source pages to check the full wording.")

if not st.session_state.messages:
    with st.container(border=True):
        st.subheader("Start with a question")
        st.write("You can ask in everyday language. The answer will include the document and page it came from.")
        st.caption("For example: How does Ebola spread? Is there a vaccine for Marburg?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_response(message["result"])

with st.container(border=True):
    with st.form("question_form", clear_on_submit=True, border=False):
        prompt = st.text_input(
            "Your question",
            placeholder="Ask about Ebola or Marburg...",
            autocomplete="off",
            disabled=not pdfs,
        )
        submitted = st.form_submit_button("Ask", type="primary", disabled=not pdfs, use_container_width=True)

if submitted and prompt.strip():
    previous_question = next(
        (message["content"] for message in reversed(st.session_state.messages) if message["role"] == "user"),
        None,
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Finding information in the documents..."):
            result = rag.ask(SimpleAssistantRequest(
                question=prompt, previous_question=previous_question, top_k=top_k, ollama_model=model,
            ))
    except Exception as exc:
        result = {"answer": f"The question could not be completed: {exc}", "error": True, "chunks": []}
    st.session_state.messages.append({"role": "assistant", "result": result})
    st.rerun()
