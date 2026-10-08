"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
    )

    if ingest_clicked and uploaded_files:
        try:
            all_chunks = []

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for uploaded_file in uploaded_files:
                    file_path = temp_path / uploaded_file.name

                    file_path.write_bytes(uploaded_file.getvalue())

                    chunks = chunker.chunk_file(file_path)
                    all_chunks.extend(chunks)

                result = store.ingest(all_chunks)

            st.session_state["last_ingestion_result"] = result

            if result.ingested > 0:
                st.sidebar.success(
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped"
                )
            elif result.skipped > 0:
                st.sidebar.warning(
                    f"No new chunks added. "
                    f"{result.skipped} duplicates skipped."
                )
            else:
                st.sidebar.warning("No chunks were added.")

            if result.errors:
                for error in result.errors:
                    st.sidebar.error(error)

            st.session_state["ingested_documents"] = [
                uploaded_file.name for uploaded_file in uploaded_files
            ]

        except Exception as exc:
            st.sidebar.error(f"Ingestion failed: {exc}")

    if st.session_state.get("ingested_documents"):
        st.sidebar.subheader("Ingested Documents")

        for document in st.session_state["ingested_documents"]:
            st.sidebar.write(f"📄 {document}")
    else:
        st.sidebar.info("Upload one or more Markdown files to populate the corpus.")

    


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    result = st.session_state.get("last_ingestion_result")

    if result is None:
        st.sidebar.metric("Total Chunks", 0)
        return

    total_chunks = result.ingested + result.skipped

    st.sidebar.metric(
        "Total Chunks",
        total_chunks,
    )


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    docs = st.session_state.get("ingested_documents", [])

    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    selected_source = st.selectbox(
        "Select document",
        options=docs,
        key="selected_document",
    )

    chunks = store.get_document_chunks(selected_source)

    if not chunks:
        st.warning("No chunks found for the selected document.")
        return

    with st.container(height=500):
        for chunk in chunks:
            metadata = chunk.metadata

            st.caption(
                f"Topic: {metadata.topic} | "
                f"Difficulty: {metadata.difficulty} | "
                f"Type: {metadata.type}"
            )

            st.write(chunk.chunk_text)
            st.divider()

    st.caption(f"{len(chunks)} chunk(s) loaded from {selected_source}")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")
    col_topic, col_diff = st.columns(2)

    with col_topic:
        topic_choice = st.selectbox(
            "Topic",
            options=["All", "LSTM", "CNN", "ANN", "GAN", "SOM", "BOLTZMANNMACHINE"],
        )
        st.session_state.topic_filter = (
            None if topic_choice == "All" else topic_choice
        )

    with col_diff:
        difficulty_choice = st.selectbox(
            "Difficulty",
            options=["All", "beginner", "intermediate", "advanced"],
        )
        st.session_state.difficulty_filter = (
            None if difficulty_choice == "All" else difficulty_choice
        )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query,
            }
        )

        graph_input = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state.topic_filter,
            "difficulty_filter": st.session_state.difficulty_filter,
        }

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
            }
        }

        try:
            result = graph.invoke(
                graph_input,
                config=config,
            )

            response = result["final_response"]

            if response is None:
                raise RuntimeError(
                    "The RAG graph did not return a final response."
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )

        except Exception as exc:
            import traceback as tb
            error_details = tb.format_exc()

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Error generating response:\n\n{error_details}",
                    "sources": [],
                    "no_context_found": False,
                }
            )

        st.rerun()


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
