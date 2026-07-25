import logging
import sqlite3
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config

logger = logging.getLogger(__name__)

# ============== OCR & Image Processing ==============

def preprocess_image_for_ocr(img):
    """Enhance image for better OCR results"""
    try:
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        img = img.filter(ImageFilter.SHARPEN)
        
        if img.size[0] < 1500 or img.size[1] < 1500:
            scale_factor = 2
            new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Upscaled image to {new_size}")
        
        return img
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise


def extract_text_with_ocr(filepath):
    """Extract text from image or PDF using OCR"""
    try:
        if filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            logger.info(f"Processing image with OCR: {filepath}")
            img = Image.open(filepath)
            logger.info(f"Original image size: {img.size}")
            
            img_processed = preprocess_image_for_ocr(img)
            
            psm_modes = [
                ('6', 'uniform block of text'),
                ('3', 'fully automatic'),
                ('4', 'single column of text'),
                ('1', 'automatic with OSD')
            ]
            
            best_text = ""
            best_mode = None
            
            for psm, desc in psm_modes:
                try:
                    text = pytesseract.image_to_string(
                        img_processed,
                        lang='eng',
                        config=f'--psm {psm} --oem 3'
                    )
                    
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        best_mode = psm
                except Exception as e:
                    logger.warning(f"PSM mode {psm} failed: {e}")
                    continue
            
            logger.info(f"Best OCR result from PSM {best_mode}: {len(best_text)} chars")
            return best_text, 1
        
        elif filepath.lower().endswith('.pdf'):
            logger.info(f"Processing PDF with OCR: {filepath}")
            
            try:
                images = convert_from_path(filepath, dpi=Config.OCR_DPI)
            except Exception as e:
                logger.warning(f"Failed at {Config.OCR_DPI} DPI, trying 200 DPI")
                images = convert_from_path(filepath, dpi=200)
            
            text = ""
            successful_pages = 0
            
            for i, image in enumerate(images):
                try:
                    img_processed = preprocess_image_for_ocr(image)
                    text += f"\n{'='*50}\nPAGE {i+1}\n{'='*50}\n\n"
                    
                    page_text = pytesseract.image_to_string(
                        img_processed,
                        lang='eng',
                        config='--psm 6 --oem 3'
                    )
                    
                    if page_text.strip():
                        text += page_text
                        successful_pages += 1
                    else:
                        text += "[No text extracted]\n"
                        
                except Exception as e:
                    logger.error(f"Error on page {i+1}: {e}")
                    text += f"[Error: {e}]\n"
            
            logger.info(f"Extracted {len(text)} chars from {successful_pages}/{len(images)} pages")
            return text, len(images)
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise


def has_meaningful_text(documents, min_chars=None):
    """Check if extracted text is meaningful"""
    if min_chars is None:
        min_chars = Config.OCR_MIN_CHARS

    if not documents:
        return False
    total_text = "".join(doc.page_content for doc in documents)
    return len(total_text.strip()) >= min_chars


def clean_extracted_text(text):
    """Clean PDF-extracted text before chunking.

    Removes:
    - TOC dot patterns (e.g., "Chapter 1 ........ 1-1")
    - "This page intentionally left blank" notices
    - Standalone page headers/footers (e.g., "ATP 2-01.3 iii")
    - Control characters (\\u0003, etc.)
    - Excessive whitespace
    """
    import re

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (will handle whitespace collapse later)
        if not stripped:
            cleaned_lines.append('')
            continue

        # Remove lines where >30% of characters are dots (TOC patterns)
        dot_count = stripped.count('.')
        if len(stripped) > 0 and dot_count / len(stripped) > 0.3:
            continue

        # Remove "intentionally left blank" notices
        if 'intentionally left blank' in stripped.lower():
            continue

        # Remove standalone page headers/footers like "ATP 2-01.3 iii" or "1 March 2019 ATP 2-01.3 v"
        if re.match(r'^[\divxlc\s\-]*ATP\s+\d[\d\-\.]*[\divxlc\s]*$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+ATP', stripped, re.IGNORECASE):
            continue

        # Strip control characters (ASCII 0-31 except newline \n=10, tab \t=9)
        cleaned = ''.join(c for c in line if ord(c) >= 32 or c in '\t')
        cleaned_lines.append(cleaned)

    # Rejoin and collapse 3+ consecutive newlines to 2
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    logger.info(f"Cleaned text: {len(text)} -> {len(result)} chars ({len(text) - len(result)} removed)")
    return result


# ============== Parent Chunk Store ==============

class ParentChunkStore:
    """SQLite-backed store for parent chunks used in hierarchical retrieval."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def add(self, parent_id, content, source, chunk_index, page=""):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO parent_chunks (parent_id, content, source, chunk_index, page) VALUES (?, ?, ?, ?, ?)",
            (parent_id, content, source, chunk_index, page)
        )
        conn.commit()
        conn.close()

    def update_page(self, parent_id, page):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE parent_chunks SET page = ? WHERE parent_id = ?",
            (page, parent_id)
        )
        conn.commit()
        conn.close()

    def get_many(self, parent_ids):
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" for _ in parent_ids)
        rows = conn.execute(
            f"SELECT parent_id, content, source, chunk_index, page FROM parent_chunks WHERE parent_id IN ({placeholders})",
            parent_ids
        ).fetchall()
        conn.close()
        return {row[0]: {"content": row[1], "source": row[2], "chunk_index": row[3], "page": row[4]} for row in rows}

    def clear(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM parent_chunks")
        conn.commit()
        conn.close()

    def delete_uploads(self):
        """Delete parent chunks for user uploads only. Knowledge Base documents
        are stored with a 'KB_' source prefix, so we keep those."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM parent_chunks WHERE source NOT LIKE 'KB\\_%' ESCAPE '\\'")
        conn.commit()
        conn.close()


# ============== Text Chunking ==============

def is_valid_chunk(text):
    """Check if a chunk has meaningful content.

    Filters out:
    - Chunks shorter than MIN_CHUNK_CHARS
    - Chunks that are mostly punctuation/whitespace (>40%)
    """
    stripped = text.strip()
    if len(stripped) < Config.MIN_CHUNK_CHARS:
        return False
    # Reject if <60% alphanumeric characters
    alnum_count = sum(c.isalnum() or c.isspace() for c in stripped)
    if alnum_count < len(stripped) * 0.6:
        return False
    return True


def create_hierarchical_chunks(text, source_name, parent_store):
    """Create parent-child chunk hierarchy for doctrine documents.

    Child chunks (small) are embedded in ChromaDB for precise semantic search.
    Parent chunks (large) are stored in SQLite and returned to the LLM when a
    child matches — giving both search precision and rich context.
    """
    # Step 1: Split into parent chunks (sent to LLM as context)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.PARENT_CHUNK_SIZE,
        chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    parent_texts = parent_splitter.split_text(text)

    # Step 2: For each parent, create child chunks and link them
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHILD_CHUNK_SIZE,
        chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )

    child_documents = []
    filtered_count = 0
    parent_count = 0

    for parent_idx, parent_text in enumerate(parent_texts):
        if not is_valid_chunk(parent_text):
            filtered_count += 1
            continue

        parent_id = f"{source_name}::parent_{parent_idx}"

        # Store parent in SQLite
        parent_store.add(parent_id, parent_text, source_name, parent_idx)
        parent_count += 1

        # Split parent into children
        child_texts = child_splitter.split_text(parent_text)

        for child_idx, child_text in enumerate(child_texts):
            if not is_valid_chunk(child_text):
                filtered_count += 1
                continue

            child_documents.append(Document(
                page_content=child_text,
                metadata={
                    "source": source_name,
                    "parent_id": parent_id,
                    "child_index": child_idx,
                    "chunk_type": "child",
                }
            ))

    logger.info(
        f"Hierarchical chunking: {parent_count} parents -> "
        f"{len(child_documents)} children (filtered {filtered_count} invalid)"
    )
    return child_documents


# ============== Search & Retrieval ==============

def similarity_search(query, vectorstore, k=10, min_score=None):
    """Semantic search with relevance score filtering.

    Returns child chunks sorted by relevance, filtered by minimum score.
    """
    if min_score is None:
        min_score = Config.MIN_RELEVANCE_SCORE

    try:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as e:
        logger.warning(f"Score-based search failed, falling back to basic search: {e}")
        return vectorstore.similarity_search(query, k=k)

    filtered = []
    for doc, score in results_with_scores:
        if score >= min_score:
            doc.metadata["relevance_score"] = round(score, 4)
            filtered.append(doc)
        else:
            logger.debug(f"Filtered chunk (score={score:.4f} < {min_score}): {doc.page_content[:80]}...")

    logger.info(f"Search: {len(results_with_scores)} results, {len(filtered)} above threshold ({min_score})")
    return filtered


def resolve_parents(child_docs, parent_store):
    """Given retrieved child documents, fetch and deduplicate their parent chunks."""
    seen_parent_ids = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen_parent_ids:
            seen_parent_ids.append(pid)

    if not seen_parent_ids:
        # Fallback: return children as-is (e.g., legacy chunks without parent_id)
        return child_docs

    parents = parent_store.get_many(seen_parent_ids)

    parent_documents = []
    for pid in seen_parent_ids:
        parent_data = parents.get(pid)
        if parent_data:
            parent_documents.append(Document(
                page_content=parent_data["content"],
                metadata={
                    "source": parent_data["source"],
                    "parent_id": pid,
                    "chunk_index": parent_data["chunk_index"],
                    "chunk_type": "parent",
                    "page": parent_data.get("page", ""),
                }
            ))

    logger.info(f"Resolved {len(child_docs)} children -> {len(parent_documents)} unique parents")
    return parent_documents


# ============== Page Annotation ==============

def _lookup_page(offset, page_offsets):
    """Given a character offset and a sorted list of (start_offset, page_num),
    return the page number that contains that offset."""
    page = page_offsets[0][1] if page_offsets else 1
    for start, pnum in page_offsets:
        if start > offset:
            break
        page = pnum
    return page


def annotate_chunks_with_pages(chunks, raw_text, page_offsets):
    """Set metadata['page'] on each chunk based on where its text appears in raw_text."""
    for chunk in chunks:
        idx = raw_text.find(chunk.page_content[:80])
        if idx >= 0:
            chunk.metadata["page"] = str(_lookup_page(idx, page_offsets))
        else:
            chunk.metadata["page"] = ""


def update_parent_pages(parent_store, raw_text, page_offsets, source_name):
    """Update page info for all parent chunks of a given source."""
    conn = __import__('sqlite3').connect(parent_store.db_path)
    rows = conn.execute(
        "SELECT parent_id, content FROM parent_chunks WHERE source = ?",
        (source_name,)
    ).fetchall()
    conn.close()

    for parent_id, content in rows:
        idx = raw_text.find(content[:80])
        if idx >= 0:
            page = str(_lookup_page(idx, page_offsets))
            parent_store.update_page(parent_id, page)


def preprocess_query(query):
    """Enhance query for better retrieval"""
    expansions = {
        "what's": "what is",
        "whats": "what is",
        "it's": "it is",
        "its": "it is",
        "dont": "do not",
        "can't": "cannot",
        "won't": "will not",
        "img": "image",
        "pic": "picture",
        "doc": "document",
        "info": "information"
    }
    
    enhanced_query = query.lower()
    for short, full in expansions.items():
        enhanced_query = enhanced_query.replace(short, full)
    
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                  'should', 'may', 'might', 'must', 'can', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during'}
    
    words = enhanced_query.split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return enhanced_query, key_terms