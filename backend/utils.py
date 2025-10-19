import logging
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
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


# ============== Text Chunking ==============

def create_smart_chunks(text, source_name):
    """Create multiple chunk sizes for better retrieval"""
    chunks = []
    
    # Small chunks (precise retrieval)
    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SMALL,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    small_chunks = small_splitter.split_text(text)
    for i, chunk in enumerate(small_chunks):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_type": "small", "chunk_id": f"small_{i}"}
        ))
    
    # Medium chunks (balanced)
    medium_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_MEDIUM,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    medium_chunks = medium_splitter.split_text(text)
    for i, chunk in enumerate(medium_chunks):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_type": "medium", "chunk_id": f"medium_{i}"}
        ))
    
    # Large chunks (context)
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_LARGE,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    large_chunks = large_splitter.split_text(text)
    for i, chunk in enumerate(large_chunks):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_type": "large", "chunk_id": f"large_{i}"}
        ))
    
    logger.info(f"Created {len(small_chunks)} small, {len(medium_chunks)} medium, {len(large_chunks)} large chunks")
    return chunks


# ============== Search & Retrieval ==============

def hybrid_search(query, vectorstore, k=5):
    """Hybrid search: vector similarity + keyword matching"""
    vector_results = vectorstore.similarity_search(query, k=k)
    
    all_docs = vectorstore.similarity_search("", k=100)
    
    query_keywords = set(query.lower().split())
    keyword_results = []
    
    for doc in all_docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_keywords & doc_words)
        if overlap > 0:
            keyword_results.append((doc, overlap))
    
    keyword_results.sort(key=lambda x: x[1], reverse=True)
    keyword_docs = [doc for doc, _ in keyword_results[:k]]
    
    combined = []
    seen_content = set()
    
    for doc in vector_results + keyword_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            combined.append(doc)
            seen_content.add(content_hash)
    
    logger.info(f"Hybrid search: {len(vector_results)} vector + {len(keyword_docs)} keyword = {len(combined)} results")
    return combined[:k * 2]


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