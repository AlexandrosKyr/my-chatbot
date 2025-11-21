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
    
    def upload_and_index(self, filepath, filename, is_kb=False):
        """Upload file, extract text, chunk, and index"""
        try:
            logger.info(f"Processing {'KB ' if is_kb else ''}document: {filename}")
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("File is empty")
            
            logger.info(f"File size: {file_size / 1024:.2f} KB")
            
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
            
            doc_entry = {
                "filename": filename,
                "content": raw_text,
                "timestamp": datetime.now().isoformat(),
                "is_kb": is_kb
            }
            self.raw_documents.append(doc_entry)
            
            chunks = create_smart_chunks(raw_text, filename)
            
            if not chunks:
                raise ValueError("Failed to create chunks")
            
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
            
            for chunk in chunks:
                chunk.metadata['is_kb'] = is_kb
            
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
            
            if os.path.exists(Config.KB_FOLDER):
                for filename in os.listdir(Config.KB_FOLDER):
                    filepath = os.path.join(Config.KB_FOLDER, filename)
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
    """Handle RAG (Retrieval-Augmented Generation) chat"""
    
    def __init__(self, llm, vectorstore, raw_documents):
        self.llm = llm
        self.vectorstore = vectorstore
        self.raw_documents = raw_documents
    
    def process_query(self, question):
        """Process chat query with multi-strategy fallback"""
        logger.info(f"Processing query: {question[:100]}...")
        
        kb_keywords = [
            'knowledge base', 'uploaded', 'documents you have', 
            'your knowledge', 'what do you know', 'do you have',
            'what documents', 'what files', 'your documents',
            'your database', 'available documents', 'kb', 'doctrine'
        ]
        
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in kb_keywords):
            kb_count = len([doc for doc in self.raw_documents if doc.get('is_kb', False)])
            regular_count = len([doc for doc in self.raw_documents if not doc.get('is_kb', False)])
            
            if kb_count > 0 or regular_count > 0:
                response = "Yes, I have access to the following documents:\n\n"
                
                if kb_count > 0:
                    response += "📚 **KNOWLEDGE BASE** (Permanent Reference Library):\n"
                    response += "Contains NATO doctrine and tactical references\n\n"
                    kb_docs = [doc for doc in self.raw_documents if doc.get('is_kb', False)]
                    for doc in kb_docs:
                        response += f"  • {doc['filename'].replace('KB_', '')} [KB]\n"
                    response += "\n"
                
                if regular_count > 0:
                    response += "📄 **TEMPORARY UPLOADS** (Current Session):\n\n"
                    regular_docs = [doc for doc in self.raw_documents if not doc.get('is_kb', False)]
                    for doc in regular_docs:
                        response += f"  • {doc['filename']} [Upload]\n"
                    response += "\n"
                
                response += "═══════════════════════════════════════════\n\n"
                response += "I can answer questions based on these documents, including:\n"
                response += "• NATO tactical doctrine and procedures\n"
                response += "• Military operational planning\n"
                response += "• Intelligence preparation of the battlefield\n"
                response += "• Strategic guidance and decision-making\n\n"
                response += "What would you like to know?"
                
                logger.info("Knowledge base query detected - listing documents")
                return response, "knowledge_base_check"
            else:
                return "I don't currently have any documents loaded. You can upload documents via the KB button (permanent) or UPLOAD button (temporary).", "knowledge_base_check"
        
        if self.vectorstore is None and len(self.raw_documents) == 0:
            logger.info("No documents, using direct LLM")
            response = self.llm.invoke(question)
            return response, "direct"
        
        try:
            logger.info("Attempting hybrid RAG")
            enhanced_query, key_terms = preprocess_query(question)

            if self.vectorstore is None:
                raise ValueError("No vector store")

            # Enhanced retrieval for doctrine: retrieve more chunks to get full context
            # Doctrine chapters are 10-15 pages, so we need broader retrieval
            retrieved_docs = hybrid_search(enhanced_query, self.vectorstore, k=10)
            
            if not retrieved_docs:
                logger.info("No documents retrieved, using direct LLM")
                response = self.llm.invoke(question)
                return response, "direct"
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            context_parts = []
            for idx, doc in enumerate(retrieved_docs, 1):
                chunk_type = doc.metadata.get('chunk_type', 'unknown')
                source = doc.metadata.get('source', 'unknown')
                is_kb = doc.metadata.get('is_kb', False)
                doc_type = '[KB - NATO Doctrine]' if is_kb else '[Upload]'
                context_parts.append(f"[Source: {source} {doc_type} | Chunk {idx} - {chunk_type}]\n{doc.page_content}")
            
            context = "\n\n" + "="*80 + "\n\n".join(context_parts)
            
            kb_notice = ""
            if any(doc.metadata.get('is_kb', False) for doc in retrieved_docs):
                kb_notice = "\n\nNOTE: Some excerpts are from the Knowledge Base containing NATO doctrine and tactical references."
            
            prompt = f"""You are a NATO tactical intelligence analyst with comprehensive access to alliance doctrine and operational guidance. Your role is to provide battlefield assessments, tactical recommendations, and strategic analysis grounded in NATO doctrine and military best practices.

IDENTITY & APPROACH:
- You operate as a NATO staff officer analyzing situations through the lens of alliance doctrine
- You are trained in NATO APP-6(E) military symbology and understand standard tactical map symbols
- Reference specific doctrine sections, chapters, and principles when applicable
- Apply military decision-making processes (MDMP) and operational planning frameworks
- Use NATO terminology, doctrinal concepts, and standard military analysis methods (infantry platoon, armor company, artillery battery, not "tank group" or "gun team")
- When doctrine is clear, cite it directly; when interpreting, explain your reasoning

DOCTRINE EXCERPTS FROM KNOWLEDGE BASE:{kb_notice}
{context}

{"="*80}

ANALYTICAL FRAMEWORK:
1. Identify relevant doctrine: Which doctrinal principles apply to this question?
2. Apply doctrine to context: How do these principles inform the specific situation?
3. Synthesize guidance: What actionable recommendations emerge from doctrine?
4. Note limitations: Where does available doctrine not fully address this scenario?

ANALYST QUERY: {question}

DOCTRINAL ASSESSMENT:"""
            
            response = self.llm.invoke(prompt)
            logger.info("Hybrid RAG successful")
            return response, "hybrid_rag"
        
        except Exception as e:
            logger.warning(f"Hybrid RAG failed: {e}")
        
        if len(self.raw_documents) > 0:
            logger.info("Falling back to raw documents")
            
            doc_sections = []
            for idx, doc in enumerate(self.raw_documents, 1):
                doc_type = '[KB - NATO Doctrine]' if doc.get('is_kb', False) else '[Upload]'
                doc_sections.append(f"[DOCUMENT {idx}: {doc['filename']} {doc_type}]\n{doc['content'][:6000]}")
            
            full_context = "\n\n" + "="*80 + "\n\n".join(doc_sections)
            
            prompt = f"""You are a NATO tactical intelligence analyst providing battlefield assessments based on alliance doctrine.

ROLE: You analyze military situations using NATO doctrinal frameworks, operational planning principles, and tactical best practices. You are NOT a generic assistant - you are a specialized NATO agent applying specific doctrinal guidance. You understand NATO APP-6(E) military symbology standards and use proper military nomenclature.

ANALYTICAL METHOD:
1. Locate relevant doctrine within the provided documents
2. Extract key principles, procedures, and guidance applicable to the query
3. Apply doctrine systematically to answer the question
4. Cite specific passages, chapters, or sections when making recommendations
5. Acknowledge when the query extends beyond available doctrine

NATO DOCTRINE & REFERENCE LIBRARY:
{full_context[:10000]}

{"="*80}

INTELLIGENCE REQUIREMENT: {question}

DOCTRINAL ANALYSIS:
[Begin by identifying which doctrine applies, then systematically apply it to address the requirement]"""
            
            response = self.llm.invoke(prompt)
            logger.info("Raw document fallback successful")
            return response, "raw_fallback"
        
        logger.warning("All RAG strategies failed, using direct LLM")
        response = self.llm.invoke(question)
        return response, "direct_fallback"


class TacticalService:
    """Handle tactical map analysis"""
    
    def __init__(self, tactical_analyzer):
        self.analyzer = tactical_analyzer
    
    def analyze_map(self, image_path, scenario, unit_types=None, vectorstore=None):
        """Run multi-model tactical analysis WITH KB"""
        if self.analyzer is None:
            raise ValueError("Tactical analyzer not available")
        
        logger.info(f"Analyzing tactical map: {scenario}")
        
        result = self.analyzer.generate_comprehensive_strategy(
            image_path,
            scenario,
            unit_types,
            vectorstore=vectorstore
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