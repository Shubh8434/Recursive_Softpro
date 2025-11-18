import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from graph_rag_pipeline import (
    GraphRAGPipeline,
    build_summary_chunks,
    build_vector_index,
    dataframe_to_chunks,
    row_to_text,
    vector_rag,
    _collect_entity_types,
)
from excel_vs_graphrag_pipeline import row_lookup, row_lookup_records


@st.cache_data(show_spinner=False)
def load_workbook(upload: bytes) -> List[Tuple[str, pd.DataFrame]]:
    """Load all non-empty sheets from an uploaded Excel workbook."""
    with io.BytesIO(upload) as buffer:
        xls = pd.ExcelFile(buffer)
        sheets: List[Tuple[str, pd.DataFrame]] = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            if not df.empty:
                sheets.append((name, df))
        return sheets


def _chunk_dataframe(task: Tuple[str, pd.DataFrame, int, str]) -> Tuple[List[Dict], int]:
    source_name, df, start_index, id_prefix = task
    return dataframe_to_chunks(
        df,
        source_name=source_name,
        start_index=start_index,
        id_prefix=id_prefix,
    )


def prepare_documents(
    sheets: List[Tuple[str, pd.DataFrame]],
    workbook_name: str | None = None,
    max_workers: int = 4,
) -> Tuple[List[str], List[Dict]]:
    documents: List[str] = []
    chunk_tasks: List[Tuple[str, pd.DataFrame, int, str]] = []
    row_index_offset = 0
    for sheet_name, df in sheets:
        sheet_docs = [row_to_text(row) for _, row in df.iterrows()]
        documents.extend(
            f"[{workbook_name}::{sheet_name}] {doc}" if workbook_name else doc
            for doc in sheet_docs
        )
        source_label = f"{workbook_name}::{sheet_name}" if workbook_name else sheet_name
        chunk_tasks.append(
            (
                source_label,
                df,
                row_index_offset,
                _safe_prefix(source_label),
            )
        )
        row_index_offset += len(sheet_docs)

    chunks: List[Dict] = []
    if chunk_tasks:
        worker_count = min(max_workers, len(chunk_tasks))
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            futures = [executor.submit(_chunk_dataframe, task) for task in chunk_tasks]
            for future in futures:
                chunk_list, _ = future.result()
                chunks.extend(chunk_list)
    return documents, chunks


def _safe_prefix(value: str) -> str:
    return value.replace(" ", "_").replace(":", "_")


def ensure_api_key(user_key: str) -> str:
    api_key = user_key.strip() or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("An OpenAI API key is required. Provide it in the sidebar or set OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


@st.cache_data(show_spinner=False)
def process_workbook(upload_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Prepare documents and chunks for a single workbook, cached by file content."""
    sheets = load_workbook(upload_bytes)
    documents, chunks = prepare_documents(sheets, workbook_name=filename)
    return {
        "sheets": sheets,
        "documents": documents,
        "chunks": chunks,
    }


def should_render_table(question: str) -> bool:
    """Heuristic to decide when to show tabular context."""
    q = question.lower()
    keyword_triggers = ("table", "tabular", "dataframe", "grid", "spreadsheet")
    if any(word in q for word in keyword_triggers):
        return True
    phrase_triggers = ("list of", "show rows", "display rows", "rows for", "list rows")
    return any(phrase in q for phrase in phrase_triggers)


def summarize_workbook(workbook: str, sheets: List[Tuple[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Compute deterministic metrics for all sheets belonging to a workbook."""
    frames = [df for _, df in sheets if not df.empty]
    if not frames:
        return {
            "workbook": workbook,
            "row_count": 0,
            "column_count": 0,
            "metrics": {},
            "top_values": {},
            "date_range": None,
            "category_metric_totals": {},
        }

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.columns = [str(col).strip() or f"column_{idx}" for idx, col in enumerate(combined.columns)]
    row_count = len(combined)
    column_count = combined.shape[1]

    metric_keywords = ("profit", "gross", "cogs", "sale", "unit", "amount", "discount", "debit", "credit")
    metrics: Dict[str, Dict[str, float]] = {}
    for column in combined.columns:
        column_lower = column.lower()
        numeric_series = pd.to_numeric(combined[column], errors="coerce")
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            continue
        if any(keyword in column_lower for keyword in metric_keywords):
            metrics[column] = {
                "total": float(numeric_series.sum()),
                "mean": float(numeric_series.mean()),
            }

    category_keywords = (
        "product",
        "segment",
        "country",
        "category",
        "verified",
        "transaction",
        "account",
        "city",
        "holder",
        "auditor",
        "status",
    )
    top_values: Dict[str, List[Tuple[str, int]]] = {}
    for column in combined.columns:
        column_lower = column.lower()
        if column in metrics:
            continue
        if any(keyword in column_lower for keyword in category_keywords):
            values = (
                combined[column]
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
            )
            if values.empty:
                continue
            counts = values.value_counts().head(5)
            if counts.empty:
                continue
            top_values[column] = [(str(index), int(count)) for index, count in counts.items()]

    category_metric_totals: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for category_column, entries in top_values.items():
        raw_values = combined[category_column].replace("", pd.NA)
        unique_count = raw_values.nunique(dropna=True)
        if unique_count and unique_count > 50:
            continue
        for metric_column, stats in metrics.items():
            numeric_series = pd.to_numeric(combined[metric_column], errors="coerce")
            pair_df = pd.DataFrame(
                {
                    "category": raw_values,
                    "metric": numeric_series,
                }
            ).dropna()
            if pair_df.empty:
                continue
            totals = (
                pair_df.groupby("category", dropna=True)["metric"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            if totals.empty:
                continue
            column_totals = category_metric_totals.setdefault(category_column, {})
            column_totals[metric_column] = [(str(index), float(value)) for index, value in totals.items()]

    date_range = None
    for column in combined.columns:
        if "date" not in column.lower():
            continue
        parsed = pd.to_datetime(combined[column], errors="coerce", utc=True)
        parsed = parsed.dropna()
        if parsed.empty:
            continue
        start = parsed.min().date().isoformat()
        end = parsed.max().date().isoformat()
        date_range = (start, end)
        break

    return {
        "workbook": workbook,
        "row_count": row_count,
        "column_count": column_count,
        "metrics": metrics,
        "top_values": top_values,
        "date_range": date_range,
        "category_metric_totals": category_metric_totals,
    }


def workbook_summary_to_chunk(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a workbook summary into a chunk for retrieval pipelines."""
    lines = [
        f"Workbook summary | file={summary['workbook']} | rows={summary['row_count']} | columns={summary['column_count']}"
    ]
    for column, stats in sorted(summary["metrics"].items()):
        lines.append(
            f"Metric | column={column} | total={stats['total']:.2f} | mean={stats['mean']:.2f}"
        )
    for column, entries in sorted(summary["top_values"].items()):
        entry_str = "; ".join(f"{value}:{count}" for value, count in entries)
        lines.append(f"Top values | column={column} | entries={entry_str}")
    for column, metrics_map in sorted(summary["category_metric_totals"].items()):
        for metric_column, totals in sorted(metrics_map.items()):
            total_entries = "; ".join(f"{value}:{amount:.2f}" for value, amount in totals)
            lines.append(
                f"Category totals | column={column} | metric={metric_column} | totals={total_entries}"
            )
    if summary["date_range"]:
        start, end = summary["date_range"]
        lines.append(f"Date range | start={start} | end={end}")

    return {
        "id": f"workbook_summary_{_safe_prefix(summary['workbook'])}",
        "text": "\n".join(lines),
        "source": summary["workbook"],
        "row_index": -1,
    }


def main():
    st.set_page_config(page_title="Excel RAG Platform", layout="wide")
    st.title("Excel RAG Playground")
    st.markdown(
        "Upload an Excel workbook, choose a retrieval strategy, and compare answers powered by GraphRAG and vector search."
    )

    with st.sidebar:
        st.header("Configuration")
        api_key_input = st.text_input("OpenAI API Key", type="password")
        max_chunks = st.slider("Graph chunks to process", min_value=10, max_value=150, value=40, step=10)

    uploaded_files = st.file_uploader(
        "Upload one or more Excel workbooks",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
    )
    question = st.text_area("Ask a question about the spreadsheet", height=120)
    retrieval_method = st.selectbox(
        "Retrieval method",
        [
            "GraphRAG (Local)",
            "GraphRAG (Global)",
            "GraphRAG (Drift)",
            "Vector RAG",
            "Row Lookup",
        ],
    )

    sheet_sources: List[Tuple[str, pd.DataFrame]] = []
    workbook_map: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    documents: List[str] = []
    base_chunks: List[Dict] = []
    if uploaded_files:
        for uploaded in uploaded_files:
            try:
                file_bytes = uploaded.getvalue()
                processed = process_workbook(file_bytes, uploaded.name)
            except Exception as exc:
                st.error(f"Failed to read {uploaded.name}: {exc}")
                continue
            sheets = processed["sheets"]
            workbook_map[uploaded.name] = sheets
            for sheet_name, df in sheets:
                label = f"{uploaded.name}::{sheet_name}"
                sheet_sources.append((label, df))
            documents.extend(processed["documents"])
            base_chunks.extend(processed["chunks"])
    else:
        st.info("Upload one or more workbooks to begin.")

    if st.button(
        "Run Retrieval",
        disabled=not sheet_sources or not question.strip(),
    ):
        if not sheet_sources:
            st.warning("Upload at least one workbook with data.")
            return

        table_required = should_render_table(question)
        try:
            api_key = ensure_api_key(api_key_input)
        except ValueError as err:
            st.error(str(err))
            return

        with st.spinner("Loading workbook(s) and preparing data..."):
            if not sheet_sources:
                st.error("No usable data found in the uploaded workbooks.")
                return
            dataframes = [df for _, df in sheet_sources]
            summary_chunks, summary_entities = build_summary_chunks(dataframes)
            workbook_summaries = []
            workbook_summary_chunks = []
            for workbook_name, sheets in workbook_map.items():
                summary = summarize_workbook(workbook_name, sheets)
                if summary["row_count"] == 0:
                    continue
                workbook_summaries.append(summary)
                workbook_summary_chunks.append(workbook_summary_to_chunk(summary))
            all_chunks = summary_chunks + workbook_summary_chunks + base_chunks
            documents_for_lookup = documents + [chunk["text"] for chunk in workbook_summary_chunks]

        if workbook_summaries:
            st.subheader("Workbook Summaries")
            for summary in workbook_summaries:
                st.markdown(f"**{summary['workbook']}**")
                st.write(f"Rows: {summary['row_count']} | Columns: {summary['column_count']}")
                if summary["metrics"]:
                    metric_rows = [
                        {
                            "Metric": column,
                            "Total": stats["total"],
                            "Average": stats["mean"],
                        }
                        for column, stats in summary["metrics"].items()
                    ]
                    metrics_df = pd.DataFrame(metric_rows)
                    st.dataframe(metrics_df.round(2))
                if summary["top_values"]:
                    top_rows = []
                    for column, entries in summary["top_values"].items():
                        for value, count in entries:
                            top_rows.append({"Column": column, "Value": value, "Count": count})
                    if top_rows:
                        st.dataframe(pd.DataFrame(top_rows))
                if summary["category_metric_totals"]:
                    agg_rows = []
                    for column, metrics_map in summary["category_metric_totals"].items():
                        for metric_column, totals in metrics_map.items():
                            for value, amount in totals:
                                agg_rows.append(
                                    {
                                        "Category Column": column,
                                        "Value": value,
                                        "Metric": metric_column,
                                        "Total": amount,
                                    }
                                )
                    if agg_rows:
                        agg_df = pd.DataFrame(agg_rows)
                        st.dataframe(agg_df.round(2))
                if summary["date_range"]:
                    start, end = summary["date_range"]
                    st.caption(f"Date range: {start} → {end}")

        with st.spinner("Initialising models..."):
            embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small",
            )
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

        if retrieval_method.startswith("GraphRAG"):
            with st.spinner("Building GraphRAG pipeline..."):
                pipeline = GraphRAGPipeline(
                    llm=llm,
                    embedding_fn=embedding_fn,
                    entity_types=_collect_entity_types(dataframes),
                )
                pipeline.refresh_graph(
                    all_chunks,
                    max_chunks=max_chunks,
                    extra_entities=summary_entities,
                )

            top_k_nodes = {"GraphRAG (Local)": 3, "GraphRAG (Global)": 5, "GraphRAG (Drift)": 2}[retrieval_method]
            max_edges = {"GraphRAG (Local)": 4, "GraphRAG (Global)": 6, "GraphRAG (Drift)": 6}[retrieval_method]

            with st.spinner("Running GraphRAG query..."):
                graph_response = pipeline.query(
                    question.strip(),
                    top_k_nodes=top_k_nodes,
                    max_edges_per_seed=max_edges,
                )
            st.subheader("GraphRAG Answer")
            st.text(graph_response)
            if table_required:
                table_records = row_lookup_records(documents_for_lookup, question.strip(), top_k=10)
                if table_records:
                    st.subheader("Relevant Rows")
                    st.dataframe(pd.DataFrame(table_records))

        elif retrieval_method == "Vector RAG":
            with st.spinner("Building vector index..."):
                vector_index = build_vector_index(all_chunks, embedding_fn)
            with st.spinner("Running vector retrieval..."):
                vector_result = vector_rag(
                    question.strip(),
                    vector_index,
                    embedding_fn,
                    llm,
                    top_k=max(6, len(all_chunks)),
                    return_context=table_required,
                )
            if table_required:
                vector_response, context_rows = vector_result
            else:
                vector_response = vector_result
                context_rows = None
            st.subheader("Vector RAG Answer")
            st.text(vector_response)
            if table_required and context_rows:
                st.subheader("Top Retrieved Chunks")
                st.dataframe(pd.DataFrame(context_rows))

        elif retrieval_method == "Row Lookup":
            st.subheader("Row Lookup")
            row_text = row_lookup(documents_for_lookup, question.strip())
            st.text(row_text)
            if table_required:
                table_records = row_lookup_records(documents_for_lookup, question.strip(), top_k=10)
                if table_records:
                    st.subheader("Matching Rows")
                    st.dataframe(pd.DataFrame(table_records))


if __name__ == "__main__":
    main()
