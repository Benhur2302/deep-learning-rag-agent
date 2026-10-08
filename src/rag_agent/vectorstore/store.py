"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            Path(self._settings.chroma_db_path).mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_db_path,
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB collection '{}' initialised with {} items",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return bool(result["ids"])

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue

                embedding = self._embeddings.embed_documents(
                    [chunk.chunk_text]
                )[0]

                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )

                result.ingested += 1

            except Exception as exc:
                result.errors.append(
                    f"{chunk.chunk_id}: {exc}"
                )

        logger.info(
            "Ingestion complete: {} ingested, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k

        where_filter = None

        if topic_filter and difficulty_filter:
            where_filter = {
                "$and": [
                    {"topic": topic_filter},
                    {"difficulty": difficulty_filter},
                ]
            }
        elif topic_filter:
            where_filter = {"topic": topic_filter}
        elif difficulty_filter:
            where_filter = {"difficulty": difficulty_filter}

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = 1 - distance

            if score < self._settings.similarity_threshold:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        retrieved_chunks.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return retrieved_chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a summary of all documents currently stored in the collection.
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        documents = {}

        for metadata in metadatas:
            if not metadata:
                continue

            source = metadata.get("source")
            if not source:
                continue

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata.get("topic", "UNKNOWN"),
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda doc: doc["source"],
        )



    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.
        """

        if not source:
            return []

        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        chunks = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(metadata),
                )
            )

        return chunks 

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        return {
            "total_chunks": self._collection.count(),
            "topics": [],
            "sources": [],
            "bonus_topics_present": False,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        results = self._collection.get(
            where={"source": source},
        )

        chunk_ids = results.get("ids") or []
        deleted_count = len(chunk_ids)

        if deleted_count > 0:
            self._collection.delete(
                where={"source": source},
            )

        return deleted_count
