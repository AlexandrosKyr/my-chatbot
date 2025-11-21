import logging
import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from config import Config
from models import get_models
from services import DocumentService, RAGService, TacticalService
from tactical_analyzer import TacticalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app_state = {
    "started_at": datetime.now().isoformat(),
    "documents_processed": 0,
    "kb_documents": 0,
    "total_queries": 0,
    "errors": 0,
    "last_error": None
}

models = get_models()
document_service = None
rag_service = None
tactical_service = None

def initialize_services():
    """Initialize all services after models are loaded"""
    global document_service, rag_service, tactical_service
    
    try:
        document_service = DocumentService(models.vectorstore)
        
        rag_service = RAGService(
            models.llm,
            models.vectorstore,
            document_service.raw_documents
        )
        
        if models.yolo_model and models.clip_model:
            analyzer = TacticalAnalyzer(
                models.llm,
                models.yolo_model,
                models.clip_model,
                models.clip_preprocess,
                models.device
            )
            tactical_service = TacticalService(analyzer)
        
        logger.info("✓ All services initialized")
    
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        logger.error(traceback.format_exc())


@app.route('/health', methods=['GET'])
def health():
    """Comprehensive health check"""
    try:
        ollama_ok, ollama_msg = models.check_ollama_connection()
        embed_ok, embed_msg = models.check_embeddings()
        is_healthy = ollama_ok and embed_ok
        
        health_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ollama": {
                    "status": "ok" if ollama_ok else "error",
                    "message": ollama_msg
                },
                "embeddings": {
                    "status": "ok" if embed_ok else "error",
                    "message": embed_msg
                },
                "vector_store": {
                    "status": "ok" if models.vectorstore is not None else "empty"
                }
            },
            "stats": {
                "documents_processed": app_state["documents_processed"],
                "kb_documents": app_state["kb_documents"],
                "total_queries": app_state["total_queries"],
                "errors": app_state["errors"]
            }
        }
        
        status_code = 200 if is_healthy else 503
        return jsonify(health_data), status_code
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    global document_service
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": "Only PDF and image files supported"}), 400
        
        if models.llm is None or models.embeddings is None:
            return jsonify({"error": "Server components not initialized"}), 500
        
        filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        result = document_service.upload_and_index(filepath, file.filename)
        app_state["documents_processed"] += 1
        
        return jsonify({
            "success": True,
            "message": f"Successfully processed {file.filename}",
            "details": result
        }), 200
    
    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload_doctrine', methods=['POST'])
def upload_doctrine():
    """Upload knowledge base documents"""
    global document_service
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        kb_filename = f"KB_{file.filename}"
        filepath = os.path.join(Config.KB_FOLDER, kb_filename)
        file.save(filepath)
        
        logger.info(f"Processing knowledge base document: {kb_filename}")
        
        result = document_service.upload_and_index(filepath, kb_filename, is_kb=True)
        app_state["kb_documents"] += 1
        
        return jsonify({
            "success": True,
            "filename": kb_filename,
            "chunks": result['chunks'],
            "text_length": result['text_length'],
            "file_size_kb": result['file_size_kb']
        }), 200
    
    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Doctrine upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/delete_all', methods=['POST'])
def delete_all():
    """Delete all documents"""
    global document_service
    
    try:
        result = document_service.delete_all()
        
        models.load_vectorstore()
        document_service = DocumentService(models.vectorstore)
        
        app_state["documents_processed"] = 0
        app_state["kb_documents"] = 0
        
        return jsonify(result), 200
    
    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat with RAG"""
    global rag_service
    
    try:
        if not request.json:
            return jsonify({"error": "Invalid request"}), 400
        
        question = request.json.get('message', '').strip()
        
        if not question:
            return jsonify({"error": "No message provided"}), 400
        
        if models.llm is None:
            return jsonify({"error": "Chat service not available"}), 503
        
        app_state["total_queries"] += 1
        
        response, mode = rag_service.process_query(question)
        
        return jsonify({
            "success": True,
            "response": response,
            "mode": mode
        }), 200
    
    except Exception as e:
        app_state["errors"] += 1
        app_state["last_error"] = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/chat",
            "error": str(e)
        }
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/analyze_tactical', methods=['POST'])
def analyze_tactical():
    """Multi-model tactical analysis"""
    global tactical_service
    
    try:
        if tactical_service is None:
            return jsonify({"error": "Tactical analyzer unavailable"}), 503
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        data = request.form
        scenario = data.get('scenario', 'Tactical operation')
        
        unit_types = None
        if 'units' in data:
            unit_types = json.loads(data['units'])
        
        filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        logger.info(f"Analysis: {file.filename}, Scenario: {scenario}")
        
        result = tactical_service.analyze_map(
            filepath, 
            scenario, 
            unit_types,
            vectorstore=models.vectorstore
        )
        
        annotated_path = os.path.join(Config.UPLOAD_FOLDER, f"annotated_{file.filename}")
        tactical_service.create_annotated_map(
            filepath,
            result['yolo_detections'],
            result['clip_regions'],
            annotated_path
        )
        
        return jsonify({
            "success": True,
            "strategy": result['strategy'],
            "models_used": result['models_used'],
            "clip_terrain": result['clip_analysis'],
            "clip_regions": result['clip_regions'],
            "yolo_detections": len(result['yolo_detections']),
            "annotated_map": f"annotated_{file.filename}"
        }), 200
    
    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/debug/chunks', methods=['GET'])
def debug_chunks():
    """Debug endpoint to see what's in vector store"""
    try:
        if models.vectorstore is None:
            return jsonify({"error": "No documents loaded", "chunks": []}), 404
        
        results = models.vectorstore.similarity_search("", k=20)
        
        chunks_info = []
        for idx, doc in enumerate(results):
            chunks_info.append({
                "index": idx,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content),
                "metadata": doc.metadata
            })
        
        return jsonify({
            "total_chunks": len(results),
            "chunks": chunks_info,
            "raw_documents": len(document_service.raw_documents) if document_service else 0
        }), 200
    
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("STARTING CHATBOT BACKEND")
    logger.info("="*60)
    logger.info(f"Debug: {Config.DEBUG}")
    logger.info(f"Models: Ollama({models.llm is not None}), YOLO({models.yolo_model is not None}), CLIP({models.clip_model is not None})")
    
    initialize_services()
    
    logger.info("="*60)
    logger.info("Starting Flask server on port 5000")
    logger.info("="*60)
    
    app.run(debug=Config.DEBUG, port=Config.PORT)