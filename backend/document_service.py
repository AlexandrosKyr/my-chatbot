import logging
import os
import re
import shutil
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from utils import (
    extract_text_with_ocr,
    has_meaningful_text,
    clean_extracted_text,
    create_hierarchical_chunks,
    annotate_chunks_with_pages,
    update_parent_pages,
)
from config import Config

logger = logging.getLogger(__name__)


class DocumentService:
    """Handle document upload, OCR, chunking, and vector store indexing."""

    def __init__(self, vectorstore, parent_store):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.raw_documents = []

    def upload_and_index(self, filepath: str, filename: str, is_kb: bool = False) -> dict:
        """Extract text from file, chunk it, and add to the vector store."""
        try:
            logger.info(f"Processing {'KB ' if is_kb else ''}document: {filename}")

            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("File is empty")

            raw_text = ""
            page_offsets = []  # list of (char_offset, page_number)

            if filepath.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents = loader.load()

                if not has_meaningful_text(documents):
                    raw_text, _ = extract_text_with_ocr(filepath)
                    for m in re.finditer(r'\n={50}\nPAGE (\d+)\n={50}\n', raw_text):
                        page_offsets.append((m.start(), int(m.group(1))))
                else:
                    parts = []
                    offset = 0
                    for doc in documents:
                        page_num = doc.metadata.get("page", 0) + 1
                        page_offsets.append((offset, page_num))
                        parts.append(doc.page_content)
                        offset += len(doc.page_content) + 2
                    raw_text = "\n\n".join(parts)
            else:
                raw_text, _ = extract_text_with_ocr(filepath)

            raw_text = clean_extracted_text(raw_text)

            if len(raw_text.strip()) < Config.OCR_MIN_CHARS:
                raise ValueError(f"Text too short (< {Config.OCR_MIN_CHARS} chars)")

            self.raw_documents.append({
                "filename": filename,
                "content": raw_text,
                "timestamp": datetime.now().isoformat(),
                "is_kb": is_kb,
            })

            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
            if self.parent_store is None:
                raise ValueError("Parent store not initialized")

            chunks = create_hierarchical_chunks(raw_text, filename, self.parent_store)
            if not chunks:
                raise ValueError("Failed to create chunks")

            if page_offsets:
                annotate_chunks_with_pages(chunks, raw_text, page_offsets)
                update_parent_pages(self.parent_store, raw_text, page_offsets, filename)

            for chunk in chunks:
                chunk.metadata["is_kb"] = is_kb

            self.vectorstore.add_documents(chunks)
            logger.info(f"Indexed {filename}: {len(chunks)} chunks")

            return {
                "success": True,
                "chunks": len(chunks),
                "text_length": len(raw_text),
                "file_size_kb": round(file_size / 1024, 2),
            }

        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise

    def delete_uploads(self) -> dict:
        """Delete only user-uploaded (non-KB) documents; keep the Knowledge Base."""
        try:
            removed_chunks = 0

            # 1. Remove non-KB chunks from the vector store.
            if self.vectorstore is not None:
                try:
                    collection = self.vectorstore._collection
                    before = collection.count()
                    collection.delete(where={"is_kb": False})
                    removed_chunks = before - collection.count()
                except Exception as e:
                    logger.error(f"Vector store upload cleanup failed: {e}")

            # 2. Delete uploaded files on disk (this folder holds only uploads).
            if os.path.exists(Config.UPLOAD_FOLDER):
                for filename in os.listdir(Config.UPLOAD_FOLDER):
                    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error deleting {filename}: {e}")

            # 3. Remove their parent chunks (KB parents are kept).
            if self.parent_store:
                self.parent_store.delete_uploads()

            # 4. Forget non-KB raw documents.
            self.raw_documents = [d for d in self.raw_documents if d.get("is_kb")]

            logger.info(f"Uploaded documents cleared ({removed_chunks} chunks)")
            return {"success": True, "chunks_deleted": removed_chunks}

        except Exception as e:
            logger.error(f"Delete uploads error: {e}")
            raise

    def delete_all(self) -> dict:
        """Delete all uploaded documents and clear the vector store."""
        try:
            if os.path.exists(Config.CHROMA_DB_PATH):
                shutil.rmtree(Config.CHROMA_DB_PATH)
                os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)

            for folder in [Config.UPLOAD_FOLDER, Config.KB_FOLDER]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                        except Exception as e:
                            logger.error(f"Error deleting {filename}: {e}")

            if self.parent_store:
                self.parent_store.clear()

            docs_deleted = len(self.raw_documents)
            self.raw_documents = []

            logger.info("All documents deleted")
            return {"success": True, "documents_deleted": docs_deleted}

        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise
