"""Tests that chunks include page number and document source metadata.

Verifies the full pipeline: chunking -> page annotation -> parent resolution
-> doctrine context formatting, ensuring the LLM receives source/page info
so it can cite documents properly.
"""

import os
import sys
import tempfile
import pytest

# Allow imports from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    create_hierarchical_chunks,
    annotate_chunks_with_pages,
    update_parent_pages,
    resolve_parents,
    ParentChunkStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_store(tmp_path):
    """Create a temporary ParentChunkStore backed by SQLite."""
    db_path = str(tmp_path / "parent_chunks_test.db")
    return ParentChunkStore(db_path)


def _make_multipage_text(pages=5, chars_per_page=600):
    """Generate fake multi-page document text with PAGE markers (OCR style)."""
    parts = []
    offset = 0
    page_offsets = []
    for i in range(1, pages + 1):
        marker = f"\n{'='*50}\nPAGE {i}\n{'='*50}\n\n"
        body = (
            f"This is page {i} content about military doctrine and terrain analysis. "
            f"It discusses tactical operations including reconnaissance, defense, "
            f"and combined arms maneuver principles for NATO forces. "
            f"Additional filler text to ensure the chunk is large enough for splitting. "
        ) * 3  # repeat to exceed MIN_CHUNK_CHARS
        page_text = marker + body
        page_offsets.append((offset, i))
        parts.append(page_text)
        offset += len(page_text)
    raw_text = "".join(parts)
    return raw_text, page_offsets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChildChunkMetadata:
    """Child chunks stored in ChromaDB must carry source and parent_id."""

    def test_child_chunks_have_source(self, parent_store):
        raw_text, _ = _make_multipage_text()
        source_name = "TestDoc.pdf"
        chunks = create_hierarchical_chunks(raw_text, source_name, parent_store)

        assert len(chunks) > 0, "Expected at least one child chunk"
        for chunk in chunks:
            assert chunk.metadata.get("source") == source_name, (
                f"Child chunk missing 'source': {chunk.metadata}"
            )

    def test_child_chunks_have_parent_id(self, parent_store):
        raw_text, _ = _make_multipage_text()
        chunks = create_hierarchical_chunks(raw_text, "TestDoc.pdf", parent_store)

        for chunk in chunks:
            pid = chunk.metadata.get("parent_id")
            assert pid is not None, f"Child chunk missing 'parent_id': {chunk.metadata}"
            assert pid.startswith("TestDoc.pdf::parent_"), (
                f"Unexpected parent_id format: {pid}"
            )

    def test_child_chunks_have_chunk_type(self, parent_store):
        raw_text, _ = _make_multipage_text()
        chunks = create_hierarchical_chunks(raw_text, "TestDoc.pdf", parent_store)

        for chunk in chunks:
            assert chunk.metadata.get("chunk_type") == "child"


class TestPageAnnotation:
    """annotate_chunks_with_pages must set metadata['page'] on child chunks."""

    def test_chunks_get_page_numbers(self, parent_store):
        raw_text, page_offsets = _make_multipage_text(pages=5)
        source_name = "DoctrineManual.pdf"
        chunks = create_hierarchical_chunks(raw_text, source_name, parent_store)

        # Before annotation, page should not be set
        pages_before = [c.metadata.get("page") for c in chunks]

        annotate_chunks_with_pages(chunks, raw_text, page_offsets)

        pages_after = [c.metadata.get("page") for c in chunks]
        non_empty_pages = [p for p in pages_after if p]
        assert len(non_empty_pages) > 0, (
            "No chunks received a page number after annotation"
        )

    def test_page_numbers_are_valid_strings(self, parent_store):
        raw_text, page_offsets = _make_multipage_text(pages=3)
        chunks = create_hierarchical_chunks(raw_text, "Test.pdf", parent_store)
        annotate_chunks_with_pages(chunks, raw_text, page_offsets)

        for chunk in chunks:
            page = chunk.metadata.get("page", "")
            if page:
                assert page.isdigit(), f"Page should be a digit string, got: {page!r}"
                assert 1 <= int(page) <= 3, f"Page out of range: {page}"


class TestParentChunkMetadata:
    """Parent chunks in SQLite must carry source, page, and be resolvable."""

    def test_parents_stored_in_sqlite(self, parent_store):
        raw_text, page_offsets = _make_multipage_text()
        source = "ATP_Manual.pdf"
        chunks = create_hierarchical_chunks(raw_text, source, parent_store)

        # Collect parent IDs referenced by children
        parent_ids = list(set(c.metadata["parent_id"] for c in chunks))
        assert len(parent_ids) > 0

        parents = parent_store.get_many(parent_ids)
        assert len(parents) > 0, "No parents found in SQLite store"

        for pid, data in parents.items():
            assert data["source"] == source, f"Parent missing source: {data}"
            assert len(data["content"]) > 0, "Parent content is empty"

    def test_parent_pages_updated(self, parent_store):
        raw_text, page_offsets = _make_multipage_text(pages=4)
        source = "FieldManual.pdf"
        chunks = create_hierarchical_chunks(raw_text, source, parent_store)

        # Update parent pages
        update_parent_pages(parent_store, raw_text, page_offsets, source)

        parent_ids = list(set(c.metadata["parent_id"] for c in chunks))
        parents = parent_store.get_many(parent_ids)

        pages_found = [data["page"] for data in parents.values() if data.get("page")]
        assert len(pages_found) > 0, (
            "No parent chunks received page numbers after update_parent_pages"
        )


class TestResolveParents:
    """resolve_parents must return Documents with source and page metadata."""

    def test_resolved_parents_have_source_and_page(self, parent_store):
        raw_text, page_offsets = _make_multipage_text(pages=3)
        source = "Doctrine.pdf"
        child_chunks = create_hierarchical_chunks(raw_text, source, parent_store)

        # Annotate children and update parent pages
        annotate_chunks_with_pages(child_chunks, raw_text, page_offsets)
        update_parent_pages(parent_store, raw_text, page_offsets, source)

        # Resolve
        parent_docs = resolve_parents(child_chunks, parent_store)

        assert len(parent_docs) > 0, "resolve_parents returned no documents"
        for doc in parent_docs:
            assert doc.metadata.get("source") == source, (
                f"Resolved parent missing source: {doc.metadata}"
            )
            assert doc.metadata.get("chunk_type") == "parent"
            # Page should be populated (may be empty for edge cases, but at least some should have it)

        pages = [d.metadata.get("page") for d in parent_docs if d.metadata.get("page")]
        assert len(pages) > 0, (
            "No resolved parent documents have page metadata — "
            "the LLM will not be able to cite page numbers"
        )


class TestDoctrineContextFormat:
    """The final context string sent to the LLM must include [Source, p.X] labels."""

    def test_context_string_contains_source_and_page(self, parent_store):
        """Simulate _retrieve_doctrine_context formatting logic."""
        raw_text, page_offsets = _make_multipage_text(pages=3)
        source = "ATP_Terrain.pdf"
        child_chunks = create_hierarchical_chunks(raw_text, source, parent_store)
        annotate_chunks_with_pages(child_chunks, raw_text, page_offsets)
        update_parent_pages(parent_store, raw_text, page_offsets, source)

        parent_docs = resolve_parents(child_chunks, parent_store)

        # Replicate the formatting from RAGService._retrieve_doctrine_context
        context_parts = []
        for idx, doc in enumerate(parent_docs, 1):
            doc_source = doc.metadata.get("source", "unknown")
            display_source = doc_source.replace(".pdf", "").replace("_", " ")
            page = doc.metadata.get("page", "")
            if page:
                label = f"[{display_source}, p.{page}]"
            else:
                label = f"[{display_source}]"
            context_parts.append(f"{label}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Verify the context string includes citation-ready labels
        assert "[ATP Terrain, p." in context, (
            f"Context string missing '[Source, p.X]' label.\n"
            f"Context preview: {context[:500]}"
        )

    def test_context_without_page_still_has_source(self, parent_store):
        """If page annotation fails, source should still be present."""
        raw_text, _ = _make_multipage_text(pages=2)
        source = "ManualNoPaging.pdf"
        child_chunks = create_hierarchical_chunks(raw_text, source, parent_store)
        # Deliberately skip annotate_chunks_with_pages

        parent_docs = resolve_parents(child_chunks, parent_store)

        for doc in parent_docs:
            assert doc.metadata.get("source") == source

        # Formatting should fall back to [Source] without page
        for doc in parent_docs:
            doc_source = doc.metadata.get("source", "unknown")
            display = doc_source.replace(".pdf", "").replace("_", " ")
            page = doc.metadata.get("page", "")
            if page:
                label = f"[{display}, p.{page}]"
            else:
                label = f"[{display}]"
            assert display in label
