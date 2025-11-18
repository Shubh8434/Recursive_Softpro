from pathlib import Path

import streamlit as st

from pipeline import (
    DEFAULT_DATASET_PATH,
    MissingCredentialError,
    build_index,
    run_query,
)

st.set_page_config(page_title="Financial Insights Q&A", layout="wide")

st.title("Financial Insights Q&A")
st.caption(
    "Interact with the LlamaIndex pipeline that parses the finance workbook, builds a "
    "vector index, and lets you question the data with different OpenAI models."
)


@st.cache_resource(show_spinner="Loading and indexing the workbook...")
def load_resources(document_path: str, llama_parse_key: str):
    index, llm_registry = build_index(document_path, llama_parse_key or None)
    return index, llm_registry


sidebar = st.sidebar
sidebar.header("Configuration")

default_path = DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else Path("")

document_path_input = sidebar.text_input(
    "Workbook path",
    value=str(default_path),
    help="Path to the .xlsx file that should be parsed and indexed.",
)

llama_parse_key = sidebar.text_input(
    "LlamaParse API key",
    type="password",
    help="Optional. Leave blank to use LLAMA_CLOUD_API_KEY / LLAMA_PARSE_API_KEY env vars.",
)

similarity_top_k = sidebar.slider(
    "Top-K nodes",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of nodes retrieved from the vector index per query.",
)

if sidebar.button("Rebuild index"):
    load_resources.clear()
    st.toast("Cache cleared. The index will rebuild on the next query.")

if not document_path_input:
    st.warning("Provide a workbook path to continue.")
    st.stop()

resolved_document_path = str(Path(document_path_input).expanduser())

if not Path(resolved_document_path).exists():
    st.error(f"Workbook not found at `{resolved_document_path}`.")
    st.stop()

try:
    index, llm_registry = load_resources(resolved_document_path, llama_parse_key)
except MissingCredentialError as exc:
    st.error(str(exc))
    st.stop()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if not isinstance(llm_registry, dict):
    st.error("The LLM registry failed to load.")
    st.stop()

model_options = list(llm_registry.keys())
selected_model = st.selectbox("Choose an OpenAI model", options=model_options, index=0)

with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Ask a question about the workbook",
        placeholder="Example: During 2014, which product registered the largest drop in monthly profit?",
        height=140,
    )
    submitted = st.form_submit_button("Run query", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a natural language question first.")
    else:
        with st.spinner(f"Querying {selected_model}..."):
            response = run_query(
                index=index,
                llm=llm_registry[selected_model],
                query=query.strip(),
                similarity_top_k=similarity_top_k,
            )

        st.subheader("Response")
        st.markdown(response.response or str(response))

        source_nodes = getattr(response, "source_nodes", None) or []
        if source_nodes:
            st.subheader("Supporting context")
            for idx, source in enumerate(source_nodes, start=1):
                metadata = source.node.metadata or {}
                label = metadata.get("file_name") or metadata.get("filename") or "Workbook"
                st.markdown(f"**Snippet {idx} — {label} (score: {source.score:.2f})**")
                st.write(source.node.get_content()[:1000])
