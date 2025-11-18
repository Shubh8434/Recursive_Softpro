"""
Utilities for running the graph-based retrieval pipeline used in the
`excel_rag_vs_graphrag_v1_large.ipynb` notebook.

The helpers below wrap the GraphRAG flow into a reusable class so the same
logic can run outside the notebook. Instantiate ``GraphRAGPipeline`` with a
LangChain-compatible chat model and embedding function, call
``refresh_graph`` with your chunked spreadsheet documents, then use
``query`` to answer questions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

GRAPH_TUPLE_DELIM = "<TUPLE>"
GRAPH_RECORD_DELIM = "<RECORD>"
GRAPH_COMPLETION_DELIM = "<END>"


class TimingTracker:
    """Utility to record durations of pipeline steps."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, float | str]] = []

    @contextmanager
    def record(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.entries.append({"label": label, "seconds": duration})
            print(f"[Timing] {label}: {duration:.2f}s", flush=True)

    def print_summary(self) -> None:
        if not self.entries:
            return
        print("\nTiming summary:", flush=True)
        for entry in self.entries:
            print(f"- {entry['label']}: {entry['seconds']:.2f}s", flush=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity for two vectors; fall back to zero on empty norms."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _slugify(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    value = value.strip("_")
    return value or "summary"


class GraphRAGPipeline:
    """Builds a knowledge graph from spreadsheet chunks and answers questions over it."""

    def __init__(
        self,
        llm,
        embedding_fn,
        entity_types: Optional[Sequence[str]] = None,
        cache_dir: Optional[Path] = Path("./graph_cache"),
        use_cache: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        llm : BaseChatModel-like
            LangChain chat model used for both extraction and answering.
        embedding_fn : Callable[[Iterable[str]], Iterable[Iterable[float]]]
            Callable that returns vector embeddings for supplied texts.
        entity_types : Sequence[str], optional
            Entity type hints supplied to the extraction prompt.
        cache_dir : Path, optional
            Directory used to persist graph extraction results between runs.
        use_cache : bool
            Toggle caching on/off. Disabled caches do not write or read disk.
        """
        self.llm = llm
        self.embedding_fn = embedding_fn
        self.default_entity_types = [et for et in (entity_types or []) if et]
        self.use_cache = use_cache and cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        extraction_prompt = ChatPromptTemplate.from_template(
            '''-Goal-
Turn the spreadsheet chunk into knowledge-graph friendly facts.

-Entity tuple format-
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

-Relationship tuple format-
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

-Instructions-
* Only use entity types from this list: {entity_types}
* entity_name should stay concise (3-5 words) and consistent across tuples.
* relationship_strength is an integer 1-10.
* Use {record_delimiter} between tuples and finish with {completion_delimiter}.

Chunk:
```
{input_text}
```

Return the tuples exactly as specified.
'''
        )

        self.graph_extraction_chain = (
            extraction_prompt.partial(
                tuple_delimiter=GRAPH_TUPLE_DELIM,
                record_delimiter=GRAPH_RECORD_DELIM,
                completion_delimiter=GRAPH_COMPLETION_DELIM,
            )
            | self.llm
            | StrOutputParser()
        )

        answer_prompt = ChatPromptTemplate.from_template(
            '''You are a helpful analyst reasoning over a spreadsheet-derived knowledge graph.

Graph context:
{context}

Question: {question}

Respond with grounded insights that reference the entities above. If the context lacks the answer, say you do not have enough information.
'''
        )

        self.graph_answer_chain = answer_prompt | self.llm | StrOutputParser()

        self.graph_entities_df: pd.DataFrame = pd.DataFrame()
        self.graph_relationships_df: pd.DataFrame = pd.DataFrame()
        self.knowledge_graph: Optional[nx.MultiDiGraph] = None
        self.node_embeddings: np.ndarray = np.empty((0, 0))
        self.node_index: Dict[str, int] = {}

    @staticmethod
    def _parse_graph_output(raw: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse the graph extraction response into entity and relationship dictionaries."""
        if not raw:
            return [], []
        clean = raw.split(GRAPH_COMPLETION_DELIM)[0]
        records = [r.strip() for r in clean.split(GRAPH_RECORD_DELIM) if r.strip()]
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        for record in records:
            if not record.startswith("(") or GRAPH_TUPLE_DELIM not in record:
                continue
            content = record.strip()[1:-1]
            parts = [p.strip().strip("'").strip('"') for p in content.split(GRAPH_TUPLE_DELIM)]
            if not parts:
                continue
            kind = parts[0].lower()
            if kind == "entity" and len(parts) >= 4:
                entities.append(
                    {
                        "entity_name": parts[1],
                        "entity_type": parts[2],
                        "entity_description": parts[3],
                    }
                )
            elif kind == "relationship" and len(parts) >= 5:
                try:
                    strength = int(parts[4])
                except ValueError:
                    strength = None
                relationships.append(
                    {
                        "source_entity": parts[1],
                        "target_entity": parts[2],
                        "relationship_description": parts[3],
                        "relationship_strength": strength,
                    }
                )
        return entities, relationships

    def _extract_graph_structure(
        self,
        chunks: Sequence[Dict[str, Any]],
        entity_types: Sequence[str],
        max_chunks: int = 20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Invoke the extraction LLM over the provided chunks and assemble graph tables."""
        seen_entities: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        relationships: List[Dict[str, Any]] = []
        entity_type_str = ", ".join(et for et in entity_types if et) or "Entity"

        for chunk in chunks[:max_chunks]:
            chunk_id = chunk.get("id", "unknown")
            # print(
            #     f"GraphRAG: extracting graph tuples from chunk {chunk_id}...",
            #     flush=True,
            # )
            chunk_text = str(chunk.get("text", ""))
            if not chunk_text.strip():
                print(f"  Skipping empty chunk {chunk_id}.", flush=True)
                continue

            cache_payload: Optional[Dict[str, Any]] = None
            cache_file: Optional[Path] = None
            if self.use_cache and self.cache_dir is not None:
                cache_key = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    try:
                        cache_payload = json.loads(cache_file.read_text())
                        print(f"  Cache hit for chunk {chunk_id}.", flush=True)
                    except json.JSONDecodeError:
                        cache_payload = None

            if cache_payload is None:
                response = self.graph_extraction_chain.invoke(
                    {
                        "entity_types": entity_type_str,
                        "input_text": chunk_text,
                    }
                )
                entities, rels = self._parse_graph_output(response)
                if self.use_cache and cache_file is not None:
                    cache_file.write_text(
                        json.dumps(
                            {
                                "entities": entities,
                                "relationships": rels,
                            }
                        )
                    )
            else:
                entities = cache_payload.get("entities", [])
                rels = cache_payload.get("relationships", [])

            for entity in entities:
                key = entity["entity_name"].strip().lower()
                if key not in seen_entities:
                    seen_entities[key] = {
                        **entity,
                        "source_doc": chunk.get("source"),
                        "row_index": chunk.get("row_index"),
                    }

            for rel in rels:
                relationships.append(
                    {
                        **rel,
                        "source_chunk": chunk.get("id"),
                        "row_index": chunk.get("row_index"),
                    }
                )
        print("GraphRAG: finished extracting tuples from chunks.", flush=True)

        entities_df = pd.DataFrame(list(seen_entities.values()))
        relationships_df = pd.DataFrame(relationships)
        return entities_df, relationships_df

    def _build_knowledge_graph(
        self,
        entities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
    ) -> Optional[nx.MultiDiGraph]:
        """Create a MultiDiGraph from the extracted entities and relationships."""
        if entities_df.empty:
            return None
        graph = nx.MultiDiGraph()

        for _, row in entities_df.iterrows():
            graph.add_node(
                row["entity_name"],
                type=row.get("entity_type", ""),
                description=row.get("entity_description", ""),
                row_index=row.get("row_index"),
            )

        if not relationships_df.empty:
            for _, row in relationships_df.iterrows():
                graph.add_edge(
                    row["source_entity"],
                    row["target_entity"],
                    description=row.get("relationship_description", ""),
                    strength=row.get("relationship_strength"),
                    source_chunk=row.get("source_chunk"),
                    row_index=row.get("row_index"),
                )

        return graph

    def refresh_graph(
        self,
        chunks: Sequence[Dict[str, Any]],
        entity_types: Optional[Sequence[str]] = None,
        max_chunks: Optional[int] = 20,
        extra_entities: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[nx.MultiDiGraph]:
        """
        Rebuild the knowledge graph from a chunk list.

        Parameters
        ----------
        chunks : Sequence[dict]
            Each chunk should provide ``text``, and ideally ``id``, ``row_index``, and ``source``.
        entity_types : Sequence[str], optional
            Override the default entity types supplied at initialization.
        max_chunks : Optional[int]
            Limit how many chunks feed the extractor to control token usage. Pass ``None`` to
            include all chunks.
        """
        if not chunks:
            self.graph_entities_df = pd.DataFrame()
            self.graph_relationships_df = pd.DataFrame()
            self.knowledge_graph = None
            self.node_embeddings = np.empty((0, 0))
            self.node_index = {}
            return None

        chunk_limit = len(chunks) if max_chunks is None else min(len(chunks), max_chunks)
        if chunk_limit < len(chunks):
            print(
                f"GraphRAG: limiting extraction to first {chunk_limit} of {len(chunks)} chunk(s).",
                flush=True,
            )
        else:
            print(f"GraphRAG: using all {len(chunks)} chunk(s) for extraction.", flush=True)

        selected_chunks = list(chunks[:chunk_limit])
        usable_chunks: List[Dict[str, Any]] = []
        chunk_hashes: List[str] = []
        for chunk in selected_chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            chunk_copy = dict(chunk)
            chunk_copy["text"] = text
            usable_chunks.append(chunk_copy)
            if self.use_cache and self.cache_dir is not None:
                chunk_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())

        if not usable_chunks:
            print("GraphRAG: no non-empty chunks available after filtering.", flush=True)
            self.graph_entities_df = pd.DataFrame()
            self.graph_relationships_df = pd.DataFrame()
            self.knowledge_graph = None
            self.node_embeddings = np.empty((0, 0))
            self.node_index = {}
            return None

        dataset_manifest_file: Optional[Path] = None
        if self.use_cache and self.cache_dir is not None and chunk_hashes:
            manifest = "|".join(chunk_hashes)
            manifest_hash = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
            dataset_manifest_file = self.cache_dir / f"{manifest_hash}_graph.json"

        types = entity_types if entity_types is not None else self.default_entity_types
        cache_loaded = False
        if dataset_manifest_file is not None and dataset_manifest_file.exists():
            try:
                data = json.loads(dataset_manifest_file.read_text())
                entities_df = pd.DataFrame(data.get("entities", []))
                relationships_df = pd.DataFrame(data.get("relationships", []))
                cache_loaded = True
                print("GraphRAG: loaded graph entities/relationships from cache.", flush=True)
            except json.JSONDecodeError:
                cache_loaded = False

        if not cache_loaded:
            print("GraphRAG: extracting entities and relationships...", flush=True)
            entities_df, relationships_df = self._extract_graph_structure(
                usable_chunks, types, max_chunks=len(usable_chunks)
            )
            print(
                f"GraphRAG: extracted {len(entities_df)} unique entities and "
                f"{len(relationships_df)} relationships.",
                flush=True,
            )
            if dataset_manifest_file is not None:
                dataset_manifest_file.write_text(
                    json.dumps(
                        {
                            "entities": entities_df.to_dict(orient="records"),
                            "relationships": relationships_df.to_dict(orient="records"),
                        }
                    )
                )

        if extra_entities:
            extra_df = pd.DataFrame(list(extra_entities))
            if not extra_df.empty:
                entities_df = pd.concat([entities_df, extra_df], ignore_index=True)
        self.graph_entities_df = entities_df
        self.graph_relationships_df = relationships_df
        print("GraphRAG: building knowledge graph...", flush=True)
        self.knowledge_graph = self._build_knowledge_graph(entities_df, relationships_df)

        if self.knowledge_graph is None or self.knowledge_graph.number_of_nodes() == 0:
            print("GraphRAG: knowledge graph is empty.", flush=True)
            self.node_embeddings = np.empty((0, 0))
            self.node_index = {}
            return self.knowledge_graph

        node_names = list(self.knowledge_graph.nodes())
        node_texts = [
            f"{name}: {self.knowledge_graph.nodes[name].get('description', '')}"
            for name in node_names
        ]
        if node_texts:
            print("GraphRAG: generating node embeddings...", flush=True)
            embeddings = self.embedding_fn(node_texts)
            self.node_embeddings = np.array(list(embeddings), dtype=float)
        else:
            self.node_embeddings = np.empty((0, 0))
        self.node_index = {name: idx for idx, name in enumerate(node_names)}
        print(
            f"GraphRAG: graph ready with {self.knowledge_graph.number_of_nodes()} nodes "
            f"and {self.knowledge_graph.number_of_edges()} edges.",
            flush=True,
        )
        return self.knowledge_graph

    def _build_graph_context(
        self,
        seed_nodes: Sequence[str],
        max_edges_per_seed: int = 4,
    ) -> str:
        """Gather node and edge facts around the seed nodes."""
        if self.knowledge_graph is None:
            return ""
        lines: List[str] = []
        for node in seed_nodes:
            if node not in self.knowledge_graph:
                continue
            data = self.knowledge_graph.nodes[node]
            lines.append(
                f"Entity: {node} (type: {data.get('type', 'unknown')}) - {data.get('description', '').strip()}"
            )
            outgoing = list(self.knowledge_graph.out_edges(node, data=True))[:max_edges_per_seed]
            for _, target, edge_data in outgoing:
                lines.append(
                    f"Forward: {node} -> {target} | {edge_data.get('description', '').strip()} (strength={edge_data.get('strength')})"
                )
            incoming = list(self.knowledge_graph.in_edges(node, data=True))[:max_edges_per_seed]
            for source, _, edge_data in incoming:
                lines.append(
                    f"Backward: {source} -> {node} | {edge_data.get('description', '').strip()} (strength={edge_data.get('strength')})"
                )
        return "\n".join(lines)

    def query(
        self,
        question: str,
        top_k_nodes: int = 3,
        max_edges_per_seed: int = 4,
    ) -> str:
        """
        Answer a question using the current knowledge graph.

        Returns a string response or a message explaining why the graph is unavailable.
        """
        if self.knowledge_graph is None or self.knowledge_graph.number_of_nodes() == 0:
            return "Knowledge graph is empty. Run `refresh_graph` first."
        if self.node_embeddings.size == 0:
            return "Node embeddings are unavailable."

        print(f"GraphRAG: running query -> {question}", flush=True)
        query_vec = np.array(self.embedding_fn([question])[0], dtype=float)
        scored: List[Tuple[float, str]] = []
        for name, idx in self.node_index.items():
            score = cosine_similarity(query_vec, self.node_embeddings[idx])
            scored.append((score, name))
        scored.sort(reverse=True)

        seed_nodes: List[str] = []
        for score, name in scored:
            if len(seed_nodes) >= top_k_nodes:
                break
            if score <= 0:
                continue
            seed_nodes.append(name)

        context = self._build_graph_context(seed_nodes, max_edges_per_seed=max_edges_per_seed)
        if not context:
            return "No contextual facts found for the graph query."

        response = self.graph_answer_chain.invoke({"context": context, "question": question})
        print("GraphRAG: query complete.", flush=True)
        return response


def row_to_text(row: pd.Series) -> str:
    """Convert a dataframe row into a concise text narration."""
    parts: List[str] = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def dataframe_to_chunks(
    df: pd.DataFrame,
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 0,
    start_index: int = 0,
    id_prefix: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return (chunk_list, document_count) for a spreadsheet dataframe."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = [row_to_text(row) for _, row in df.iterrows()]
    documents = [doc for doc in documents if doc.strip()]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    safe_prefix = (id_prefix or source_name).replace(" ", "_").replace(":", "_")

    chunks: List[Dict[str, Any]] = []
    for doc_id, doc in enumerate(documents):
        row_number = start_index + doc_id
        for chunk_id, chunk in enumerate(text_splitter.split_text(doc)):
            chunks.append(
                {
                    "id": f"{safe_prefix}_chunk{row_number}_{chunk_id}",
                    "text": chunk,
                    "source": source_name,
                    "row_index": row_number,
                }
            )
    return chunks, len(documents)


def _list_excel_files(upload_dir: Path) -> List[Path]:
    return sorted(
        upload_dir.glob("*.xls*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _collect_entity_types(frames: Sequence[pd.DataFrame]) -> List[str]:
    unique_types = set()
    for df in frames:
        unique_types.update(str(col).strip() for col in df.columns if str(col).strip())
    return sorted(unique_types)


def tabular_lookup(question: str, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return records whose string values appear in the question text."""
    question_lower = question.lower()
    hits: List[Dict[str, Any]] = []
    for entry in records:
        data = entry.get("data", {})
        for value in data.values():
            if isinstance(value, str) and value.strip() and value.lower() in question_lower:
                hits.append(entry)
                break
    return hits


def format_record(entry: Dict[str, Any]) -> str:
    data = entry.get("data", {})
    source = entry.get("source", "unknown")
    row_index = entry.get("row_index", "N/A")
    lines = [f"Source: {source} (row {row_index})"]
    for key, value in data.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def build_summary_chunks(
    dataframes: Sequence[pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create high-level summary chunks and structured entities without schema assumptions."""
    if not dataframes:
        return [], []

    combined = pd.concat(dataframes, ignore_index=True)
    combined.columns = [str(col).strip() or f"column_{idx}" for idx, col in enumerate(combined.columns)]

    total_rows = len(combined)
    total_columns = len(combined.columns)

    summary_chunks: List[Dict[str, Any]] = []
    summary_entities: List[Dict[str, Any]] = [
        {
            "entity_name": "Dataset Row Count",
            "entity_type": "SummaryMetric",
            "entity_description": f"Dataset contains {total_rows} row(s) and {total_columns} column(s).",
        }
    ]

    overview_chunk = {
        "id": "summary_overview",
        "text": f"Dataset summary | rows={total_rows} | columns={total_columns}",
        "source": "summary",
        "row_index": -1,
    }
    summary_chunks.append(overview_chunk)

    # Numeric columns
    numeric_series: Dict[str, pd.Series] = {}
    for col in combined.columns:
        coerced = pd.to_numeric(combined[col], errors="coerce")
        if coerced.notna().sum() > 0:
            numeric_series[col] = coerced

    numeric_columns = [col for col in combined.columns if col in numeric_series]
    numeric_columns.sort()

    for col in numeric_columns:
        series = numeric_series[col].dropna()
        if series.empty:
            continue
        count = int(series.count())
        total = float(series.sum())
        mean = float(series.mean())
        min_val = float(series.min())
        max_val = float(series.max())
        chunk_id = f"summary_numeric_{_slugify(col)}"
        chunk_lines = [
            f"Numeric summary | column={col} | count={count} | total={total:.2f} | mean={mean:.2f} | min={min_val:.2f} | max={max_val:.2f}"
        ]
        summary_chunks.append(
            {
                "id": chunk_id,
                "text": "\n".join(chunk_lines),
                "source": "summary",
                "row_index": -1,
            }
        )
        summary_entities.extend(
            [
                {
                    "entity_name": f"{col} Total",
                    "entity_type": "NumericTotal",
                    "entity_description": f"{col} total is {total:.2f}.",
                },
                {
                    "entity_name": f"{col} Mean",
                    "entity_type": "NumericMean",
                    "entity_description": f"{col} mean is {mean:.2f}.",
                },
                {
                    "entity_name": f"{col} Min",
                    "entity_type": "NumericMin",
                    "entity_description": f"{col} minimum is {min_val:.2f}.",
                },
                {
                    "entity_name": f"{col} Max",
                    "entity_type": "NumericMax",
                    "entity_description": f"{col} maximum is {max_val:.2f}.",
                },
            ]
        )

    # Categorical columns
    categorical_cols = [col for col in combined.columns if col not in numeric_series]
    cat_info: List[Tuple[str, int]] = []
    for col in categorical_cols:
        values = combined[col].dropna()
        unique_count = values.nunique()
        if unique_count == 0:
            continue
        cat_info.append((col, unique_count))

    cat_info.sort(key=lambda x: x[1])
    selected_categories: List[str] = []
    for col, uniq in cat_info:
        if len(selected_categories) < 5:
            selected_categories.append(col)
        elif uniq <= 30 and col not in selected_categories and len(selected_categories) < 10:
            selected_categories.append(col)
        if len(selected_categories) >= 10:
            break

    numeric_for_metrics = numeric_columns[:5]
    for col in selected_categories:
        values = combined[col].dropna()
        if values.empty:
            continue
        value_counts = values.astype(str).value_counts()
        top_values = value_counts.head(20)
        chunk_lines = [
            f"Category summary | column={col} | unique={value_counts.size} | top_values={len(top_values)}"
        ]
        for value, count in top_values.items():
            chunk_lines.append(f"Category value | column={col} | value={value} | count={int(count)}")

        for num_col in numeric_for_metrics:
            numeric_values = numeric_series.get(num_col)
            if numeric_values is None:
                continue
            grouped = numeric_values.groupby(combined[col]).sum()
            grouped = grouped.dropna().sort_values(ascending=False)
            if grouped.empty:
                continue
            top_items = list(grouped.items())[: min(10, len(grouped))]
            totals_line = "; ".join(f"{label}:{value:.2f}" for label, value in top_items)
            chunk_lines.append(
                f"Category totals | column={col} | metric={num_col} | values={totals_line}"
            )
            summary_line_entries = "; ".join(
                f"{label}:{value:.2f}@rank{rank}" for rank, (label, value) in enumerate(top_items, start=1)
            )
            chunk_lines.append(
                f"Category ranking summary | column={col} | metric={num_col} | entries={summary_line_entries}"
            )
            for rank, (label, value) in enumerate(top_items, start=1):
                label_str = str(label)
                summary_entities.append(
                    {
                        "entity_name": f"{col} rank {rank} ({num_col})",
                        "entity_type": "CategoryRanking",
                        "entity_description": f"Rank {rank} for {col} by {num_col} is {label_str} with {value:.2f}.",
                    }
                )
                summary_entities.append(
                    {
                        "entity_name": f"{col}={label_str} ({num_col})",
                        "entity_type": "CategoryMetric",
                        "entity_description": f"{col}={label_str} has {num_col} total {value:.2f}.",
                    }
                )

        mode_value = top_values.index[0]
        mode_count = int(top_values.iloc[0])
        chunk_lines.append(f"Category mode | column={col} | value={mode_value} | count={mode_count}")

        summary_entities.append(
            {
                "entity_name": f"{col} Mode",
                "entity_type": "CategorySummary",
                "entity_description": f"Most frequent value for {col} is {mode_value} appearing {mode_count} time(s).",
            }
        )

        summary_chunks.append(
            {
                "id": f"summary_category_{_slugify(col)}",
                "text": "\n".join(chunk_lines),
                "source": "summary",
                "row_index": -1,
            }
        )

    return summary_chunks, summary_entities


def build_vector_index(
    chunks: Sequence[Dict[str, Any]],
    embedding_fn,
) -> Optional[Dict[str, Any]]:
    """Pre-compute embeddings for traditional vector RAG."""
    if not chunks:
        return None
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings_list = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedding_fn(batch)
        embeddings_list.extend(batch_embeddings)
    embeddings = np.array(embeddings_list, dtype=float)
    return {
        "texts": texts,
        "embeddings": embeddings,
        "chunks": list(chunks),
    }


def vector_rag(
    question: str,
    index: Dict[str, Any],
    embedding_fn,
    llm,
    top_k: int = 4,
    return_context: bool = False,
) -> str | Tuple[str, List[Dict[str, Any]]]:
    """Answer using simple vector retrieval over chunk texts."""
    if not index or not index.get("texts"):
        return "Vector index is empty."
    query_vec = np.array(embedding_fn([question])[0], dtype=float)
    scored = []
    for idx, chunk in enumerate(index["chunks"]):
        if str(chunk.get("id", "")).startswith("summary"):
            # Guarantee summary chunks appear first
            boost = 2.0
        else:
            boost = 1.0
        score = boost * cosine_similarity(query_vec, index["embeddings"][idx])
        scored.append((score, idx))
    scored.sort(reverse=True)
    summary_indices = [idx for _, idx in scored if str(index["chunks"][idx].get("id", "")).startswith("summary")]
    non_summary_indices = [idx for _, idx in scored if idx not in summary_indices]
    max_summary = min(len(summary_indices), 10)
    summary_indices = summary_indices[:max_summary]
    needed = max(top_k, len(summary_indices))
    picked = summary_indices + [idx for idx in non_summary_indices if idx not in summary_indices][: max(0, needed - len(summary_indices))]
    top = picked
    context_lines = []
    context_rows: List[Dict[str, Any]] = []
    score_lookup = {idx: score for score, idx in scored}
    for idx in top:
        chunk = index["chunks"][idx]
        preview = index["texts"][idx]
        chunk_id = chunk.get("id", idx)
        source = chunk.get("source")
        context_lines.append(
            f"Chunk {chunk_id} (source={source}): {preview}"
        )
        context_rows.append(
            {
                "id": chunk_id,
                "source": source,
                "score": score_lookup.get(idx, 0.0),
                "text": preview,
            }
        )
    context = "\n\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful financial analyst. Use the snippets below to answer the question.

Snippets:
{context}

Question: {question}

Respond concisely and cite any relevant chunk IDs."""
    )
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = getattr(response, "content", str(response))
    if return_context:
        return answer, context_rows
    return answer


if __name__ == "__main__":
    import os
    import sys

    try:
        from chromadb.utils import embedding_functions
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - runtime guidance
        print("Missing dependencies needed for the demo run:", exc)
        sys.exit(1)

    uploads_dir = Path("./uploaded_excels")
    if not uploads_dir.exists():
        print("No 'uploaded_excels' directory found. Upload a spreadsheet or adjust the path.")
        sys.exit(0)

    print("Step 1/6: scanning for spreadsheets in 'uploaded_excels'...", flush=True)
    excel_paths = _list_excel_files(uploads_dir)
    if not excel_paths:
        print("No Excel files located in 'uploaded_excels'. Add a file and rerun.")
        sys.exit(0)
    print(f"Step 1/6: found {len(excel_paths)} spreadsheet(s).", flush=True)

    tracker = TimingTracker()

    dataframes: List[pd.DataFrame] = []
    df_sources: List[str] = []
    total_chunks: List[Dict[str, Any]] = []
    table_records: List[Dict[str, Any]] = []
    global_row_counter = 0

    for idx, excel_path in enumerate(excel_paths, start=1):
        print(f"Step 2/6: loading spreadsheet {idx}/{len(excel_paths)} -> {excel_path}", flush=True)
        with tracker.record(f"Load workbook {excel_path.name}"):
            sheet_map = pd.read_excel(excel_path, sheet_name=None)
        for sheet_name, sheet_df in sheet_map.items():
            if sheet_df.empty:
                print(
                    f"  Skipping empty sheet '{sheet_name}' in {excel_path.name}.",
                    flush=True,
                )
                continue
            source_label = f"{excel_path.name}::{sheet_name}"
            dataframes.append(sheet_df)
            df_sources.append(source_label)
            print(f"  Step 3/6: chunking sheet '{sheet_name}'...", flush=True)
            with tracker.record(f"Chunk sheet {source_label}"):
                chunks, doc_count = dataframe_to_chunks(
                    sheet_df,
                    source_label,
                    start_index=global_row_counter,
                    id_prefix=f"{excel_path.stem}_{sheet_name}",
                )
            global_row_counter += doc_count
            total_chunks.extend(chunks)
            print(
                f"  Step 3/6: sheet '{sheet_name}' chunked into {len(chunks)} chunk(s).",
                flush=True,
            )
            for row_idx, row in enumerate(sheet_df.to_dict(orient="records")):
                table_records.append(
                    {
                        "source": source_label,
                        "row_index": row_idx,
                        "data": row,
                    }
                )

    if not total_chunks:
        print("No textual chunks generated from the spreadsheets.")
        sys.exit(0)
    print(f"Step 3/6: chunking complete ({len(total_chunks)} chunks across all sheets).", flush=True)

    summary_chunks, summary_entities = build_summary_chunks(dataframes)
    if summary_chunks:
        total_chunks = summary_chunks + total_chunks
        print(
            f"Prepended {len(summary_chunks)} aggregate summary chunk(s) for GraphRAG grounding.",
            flush=True,
        )
    else:
        summary_entities = []

    print("Step 4/6: checking OpenAI API key...", flush=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set the OPENAI_API_KEY environment variable before running the demo.")
        sys.exit(0)

    print("Step 5/6: preparing embedding function and chat model...", flush=True)
    with tracker.record("Initialise embeddings & LLM"):
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("Step 6/6: building knowledge graph and answering question...", flush=True)
    pipeline = GraphRAGPipeline(
        llm=llm,
        embedding_fn=embedding_fn,
        entity_types=_collect_entity_types(dataframes),
    )
    with tracker.record("Refresh graph (max_chunks=30)"):
        pipeline.refresh_graph(total_chunks, max_chunks=30, extra_entities=summary_entities)

    vector_index = None
    with tracker.record("Build vector RAG index"):
        vector_index = build_vector_index(total_chunks, embedding_fn)

    question = "Which month delivers the highest total profit, and how much?"
    print(f"\nQuestion: {question}")

    mode_settings = [
        ("Local", {"top_k_nodes": 3, "max_edges_per_seed": 4}),
        ("Global", {"top_k_nodes": 5, "max_edges_per_seed": 6}),
        ("Drift", {"top_k_nodes": 2, "max_edges_per_seed": 6}),
    ]

    for label, kwargs in mode_settings:
        print(f"\nGraphRAG mode: {label} -> params={kwargs}", flush=True)
        with tracker.record(f"Run query ({label})"):
            answer = pipeline.query(question, **kwargs)
        print(f"{label} response:\n{answer}\n", flush=True)

    if vector_index:
        print("Vector RAG response:", flush=True)
        with tracker.record("Run vector RAG"):
            vector_answer = vector_rag(question, vector_index, embedding_fn, llm, top_k=4)
        print(vector_answer, flush=True)

    table_hits = tabular_lookup(question, table_records)
    if table_hits:
        print("\nRow lookup matches:", flush=True)
        for entry in table_hits[:3]:
            print(format_record(entry), flush=True)

    tracker.print_summary()
