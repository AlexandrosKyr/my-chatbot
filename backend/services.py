import logging
import os
import shutil
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from utils import (
    extract_text_with_ocr, 
    has_meaningful_text, 
    create_smart_chunks,
    hybrid_search,
    preprocess_query
)
from config import Config

logger = logging.getLogger(__name__)


class DocumentService:
    """Handle document upload, OCR, and indexing"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.raw_documents = []
    
    def upload_and_index(self, filepath, filename):
        """Upload file, extract text, chunk, and index"""
        try:
            logger.info(f"Processing document: {filename}")
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("File is empty")
            
            logger.info(f"File size: {file_size / 1024:.2f} KB")
            
            # Extract text
            raw_text = ""
            if filepath.lower().endswith('.pdf'):
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                
                if not has_meaningful_text(documents):
                    ocr_text, _ = extract_text_with_ocr(filepath)
                    raw_text = ocr_text
                else:
                    raw_text = "\n\n".join([doc.page_content for doc in documents])
            else:
                ocr_text, _ = extract_text_with_ocr(filepath)
                raw_text = ocr_text
            
            if len(raw_text.strip()) < Config.OCR_MIN_CHARS:
                raise ValueError(f"Text too short (< {Config.OCR_MIN_CHARS} chars)")
            
            # Store raw text for fallback
            self.raw_documents.append({
                "filename": filename,
                "content": raw_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Create chunks and index
            chunks = create_smart_chunks(raw_text, filename)
            
            if not chunks:
                raise ValueError("Failed to create chunks")
            
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
            
            self.vectorstore.add_documents(chunks)
            
            logger.info(f"Successfully indexed {filename}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "chunks": len(chunks),
                "text_length": len(raw_text),
                "file_size_kb": round(file_size / 1024, 2)
            }
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise
    
    def delete_all(self):
        """Delete all documents and reset"""
        try:
            if os.path.exists(Config.CHROMA_DB_PATH):
                shutil.rmtree(Config.CHROMA_DB_PATH)
                os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
            
            if os.path.exists(Config.UPLOAD_FOLDER):
                for filename in os.listdir(Config.UPLOAD_FOLDER):
                    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error deleting {filename}: {e}")
            
            docs_deleted = len(self.raw_documents)
            self.raw_documents = []
            
            logger.info("All documents deleted")
            return {"success": True, "documents_deleted": docs_deleted}
        
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise


class RAGService:
    """Handle RAG chat"""
    
    def __init__(self, llm, vectorstore, raw_documents):
        self.llm = llm
        self.vectorstore = vectorstore
        self.raw_documents = raw_documents
    
    def process_query(self, question):
        """Process chat query with multi-strategy fallback"""
        logger.info(f"Processing query: {question[:100]}...")
        
        # Strategy 1: No documents - direct LLM
        if self.vectorstore is None and len(self.raw_documents) == 0:
            logger.info("No documents, using direct LLM")
            response = self.llm.invoke(question)
            return response, "direct"
        
        # Strategy 2: Hybrid RAG
        try:
            logger.info("Attempting hybrid RAG")
            enhanced_query, key_terms = preprocess_query(question)
            retrieved_docs = hybrid_search(enhanced_query, self.vectorstore, k=7)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            context_parts = []
            for idx, doc in enumerate(retrieved_docs, 1):
                chunk_type = doc.metadata.get('chunk_type', 'unknown')
                context_parts.append(f"[Document Chunk {idx} - {chunk_type}]\n{doc.page_content}")
            
            context = "\n\n" + "="*80 + "\n\n".join(context_parts)
            
            prompt = f"""You are reading text extracted from an image or document using OCR.

CRITICAL: If asked "what does this say" or "what text is in this image", extract and present the actual text content from below.

DOCUMENT TEXT:
{context}

{"="*80}

USER QUESTION: {question}

ANSWER (extract relevant text or answer the question):"""
            
            response = self.llm.invoke(prompt)
            logger.info("Hybrid RAG successful")
            return response, "hybrid_rag"
        
        except Exception as e:
            logger.warning(f"Hybrid RAG failed: {e}")
        
        # Strategy 3: Fallback to raw documents
        if len(self.raw_documents) > 0:
            logger.info("Falling back to raw documents")
            
            doc_sections = []
            for idx, doc in enumerate(self.raw_documents, 1):
                doc_sections.append(f"[DOCUMENT {idx}: {doc['filename']}]\n{doc['content'][:6000]}")
            
            full_context = "\n\n" + "="*80 + "\n\n".join(doc_sections)
            
            prompt = f"""You are a helpful AI analyzing documents. Answer based on the documents below.

INSTRUCTIONS:
1. Read the entire document carefully
2. Provide a comprehensive answer
3. If not in the document, say so clearly
4. Be specific and reference relevant parts
5. Do not make assumptions beyond the text, if unsure then ask for clarification

FULL DOCUMENT(S):
{full_context[:10000]}

{"="*80}

USER QUESTION: {question}

ANSWER:"""
            
            response = self.llm.invoke(prompt)
            logger.info("Raw document fallback successful")
            return response, "raw_fallback"
        
        # Strategy 4: Last resort
        logger.warning("All RAG strategies failed, using direct LLM")
        response = self.llm.invoke(question)
        return response, "direct_fallback"


class TacticalService:
    """Handle tactical map analysis"""
    
    def __init__(self, tactical_analyzer):
        self.analyzer = tactical_analyzer
    
    def analyze_map(self, image_path, scenario, unit_types=None):
        """Run multi-model tactical analysis"""
        if self.analyzer is None:
            raise ValueError("Tactical analyzer not available")
        
        logger.info(f"Analyzing tactical map: {scenario}")
        
        result = self.analyzer.generate_comprehensive_strategy(
            image_path,
            scenario,
            unit_types
        )
        
        logger.info("Tactical analysis complete")
        return result
    
    def create_annotated_map(self, image_path, yolo_detections, clip_regions, output_path):
        """Create annotated map visualization"""
        if self.analyzer is None:
            raise ValueError("Tactical analyzer not available")
        
        return self.analyzer.create_annotated_map(
            image_path,
            yolo_detections,
            clip_regions,
            output_path
        )