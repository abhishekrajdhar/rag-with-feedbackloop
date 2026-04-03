"""Streamlit frontend for the production RAG service."""

from __future__ import annotations

import json
import os
from typing import Dict, Generator, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
REQUEST_TIMEOUT = 180


def initialize_state() -> None:
    """Set up session state containers."""
    st.session_state.setdefault("latest_query", None)
    st.session_state.setdefault("feedback_target_id", "")
    st.session_state.setdefault("feedback_result", None)
    st.session_state.setdefault("streamed_answer", "")
    st.session_state.setdefault("stream_metadata", None)


def api_healthcheck() -> tuple[bool, str]:
    """Return whether the backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return True, payload.get("status", "ok")
    except requests.RequestException as exc:
        return False, str(exc)


def ingest_documents(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Dict:
    """Upload documents to the backend ingestion endpoint."""
    files = []
    for uploaded_file in uploaded_files:
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                ),
            )
        )

    response = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def query_documents(question: str, top_k: int) -> Dict:
    """Run a standard query against the backend."""
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def stream_query(question: str, top_k: int) -> Generator[str, None, Dict]:
    """Stream an answer from the SSE query endpoint."""
    response = requests.post(
        f"{API_BASE_URL}/query/stream",
        json={"question": question, "top_k": top_k},
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    current_event: Optional[str] = None
    latest_text = ""
    final_payload: Optional[Dict] = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue

        data = line.split(":", 1)[1].strip()
        payload = json.loads(data)

        if current_event == "metadata":
            st.session_state["stream_metadata"] = payload
            continue
        if current_event == "token":
            latest_text += payload.get("text", "")
            yield latest_text
            continue
        if current_event == "error":
            raise requests.HTTPError(payload.get("detail", "Streaming request failed."))
        if current_event == "done":
            final_payload = payload
            break

    if final_payload is None:
        raise RuntimeError("Streaming response ended without a final payload.")
    return final_payload


def submit_feedback(query_id: str, rating: int, correction: str) -> Dict:
    """Submit feedback to the backend."""
    payload = {"query_id": query_id, "rating": rating, "correction": correction or None}
    response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_mlflow_summary() -> Dict:
    """Fetch the MLflow summary from the backend."""
    response = requests.get(f"{API_BASE_URL}/mlflow/summary", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def render_query_result(result: Dict) -> None:
    """Render a query response payload."""
    st.subheader("Answer")
    st.write(result["answer"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("Latency", f"{result['latency']:.2f}s")
    metric_columns[1].metric("Hallucination", f"{result['hallucination_score']:.3f}")
    metric_columns[2].metric("Verdict", result["verdict"])
    metric_columns[3].metric("Retried", "Yes" if result["retried"] else "No")

    st.caption(f"Query ID: {result['query_id']}")

    with st.expander("Retrieved Sources", expanded=True):
        for source, score in zip(result["sources"], result["retrieval_scores"]):
            st.markdown(f"**{source['source_file']}**  |  `{source['chunk_id']}`  |  score `{score['score']:.4f}`")
            st.write(source["content"])
            st.divider()


def render_streaming_panel(question: str, top_k: int) -> None:
    """Run and render a streaming query."""
    placeholder = st.empty()
    metadata_placeholder = st.empty()
    st.session_state["stream_metadata"] = None

    try:
        generator = stream_query(question, top_k)
        final_result = None
        while True:
            try:
                partial_answer = next(generator)
                placeholder.markdown(partial_answer)
                if st.session_state["stream_metadata"]:
                    metadata_placeholder.info("Retrieved context loaded. Streaming answer in progress.")
            except StopIteration as stop:
                final_result = stop.value
                break

        if final_result is None:
            raise RuntimeError("No final streaming result was returned.")

        st.session_state["latest_query"] = final_result
        st.session_state["feedback_target_id"] = final_result["query_id"]
        placeholder.empty()
        metadata_placeholder.empty()
        render_query_result(final_result)
    except Exception as exc:
        st.error(f"Streaming query failed: {exc}")


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="RAG Control Center", page_icon="📚", layout="wide")
    initialize_state()
    st.title("RAG Control Center")
    st.caption("Ingest documents, run grounded queries, review evidence, and push corrective feedback.")

    with st.sidebar:
        st.header("Service")
        st.code(API_BASE_URL)
        healthy, message = api_healthcheck()
        if healthy:
            st.success(f"Backend status: {message}")
        else:
            st.error(f"Backend unavailable: {message}")
        st.link_button("Open MLflow UI", MLFLOW_UI_URL)

    ingest_tab, query_tab, feedback_tab, tracking_tab = st.tabs(
        ["Ingest", "Query", "Feedback", "Tracking"]
    )

    with ingest_tab:
        st.subheader("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one file first.")
            else:
                try:
                    with st.spinner("Uploading and indexing documents..."):
                        result = ingest_documents(uploaded_files)
                    st.success(
                        f"Indexed {result['chunks_indexed']} chunks from {result['files_processed']} files."
                    )
                    st.json(result)
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with query_tab:
        st.subheader("Ask the RAG System")
        question = st.text_area(
            "Question",
            placeholder="Ask a question grounded in the ingested documents...",
            height=120,
        )
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=4)
        streaming = st.toggle("Use streaming query", value=False)

        if st.button("Run Query", type="primary", use_container_width=True):
            if len(question.strip()) < 3:
                st.warning("Enter a longer question before querying.")
            else:
                if streaming:
                    render_streaming_panel(question.strip(), top_k)
                else:
                    try:
                        with st.spinner("Retrieving context and generating answer..."):
                            result = query_documents(question.strip(), top_k)
                        st.session_state["latest_query"] = result
                        st.session_state["feedback_target_id"] = result["query_id"]
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

        if st.session_state["latest_query"]:
            st.divider()
            render_query_result(st.session_state["latest_query"])

    with feedback_tab:
        st.subheader("Feedback Loop")
        default_query_id = st.session_state.get("feedback_target_id", "")
        query_id = st.text_input("Query ID", value=default_query_id)
        rating = st.slider("Rating", min_value=1, max_value=5, value=3)
        correction = st.text_area(
            "Correction",
            placeholder="Optional. Required if rating is 1 or 2 and you want corrective memory added.",
            height=120,
        )

        if st.button("Submit Feedback", use_container_width=True):
            if not query_id.strip():
                st.warning("Provide a query ID first.")
            else:
                try:
                    with st.spinner("Submitting feedback..."):
                        result = submit_feedback(query_id.strip(), rating, correction.strip())
                    st.session_state["feedback_result"] = result
                    st.success("Feedback stored successfully.")
                except Exception as exc:
                    st.error(f"Feedback failed: {exc}")

        if st.session_state["feedback_result"]:
            result = st.session_state["feedback_result"]
            st.json(result)
            if result.get("improved_answer"):
                st.subheader("Improved Answer")
                st.write(result["improved_answer"])

    with tracking_tab:
        st.subheader("MLflow Summary")
        if st.button("Refresh Tracking", use_container_width=True):
            st.rerun()
        try:
            summary = fetch_mlflow_summary()
            metric_columns = st.columns(3)
            metric_columns[0].metric("Total Runs", summary["total_runs"])
            metric_columns[1].metric("Avg Latency", f"{summary['average_latency']:.2f}s")
            metric_columns[2].metric(
                "Avg Hallucination",
                f"{summary['average_hallucination_score']:.3f}",
            )

            st.write("Verdict counts")
            st.json(summary["verdict_counts"])

            if summary["recent_runs"]:
                st.write("Recent runs")
                st.dataframe(summary["recent_runs"], use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load MLflow summary: {exc}")


if __name__ == "__main__":
    main()
