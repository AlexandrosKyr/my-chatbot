import logging
import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
from config import Config
from models import get_models
from services import DocumentService, RAGService
from utils import ParentChunkStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload limit
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

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

def initialize_services():
    """Initialize all services after models are loaded"""
    global document_service, rag_service

    try:
        document_service = DocumentService(models.vectorstore, models.parent_store)

        rag_service = RAGService(
            models.llm,
            models.vectorstore,
            document_service.raw_documents,
            models.parent_store
        )

        logger.info("All services initialized (RAGService handles terrain-aware tactical analysis)")

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
        return jsonify({"status": "error", "message": "Health check failed"}), 500


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

        # Sanitize filename to prevent path traversal attacks
        safe_filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
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
        return jsonify({"error": "Failed to process uploaded file"}), 500


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

        # Sanitize filename to prevent path traversal attacks
        safe_filename = secure_filename(file.filename)
        kb_filename = f"KB_{safe_filename}"
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
        return jsonify({"error": "Failed to process doctrine document"}), 500


@app.route('/delete_all', methods=['POST'])
def delete_all():
    """Delete all documents. Requires {"confirm": true} in the request body."""
    global document_service

    try:
        if not request.json or not request.json.get('confirm'):
            return jsonify({"error": "Must send {\"confirm\": true} to delete all data"}), 400

        result = document_service.delete_all()

        models.load_vectorstore()
        models.parent_store = ParentChunkStore(Config.PARENT_CHUNKS_DB_PATH)
        document_service = DocumentService(models.vectorstore, models.parent_store)

        app_state["documents_processed"] = 0
        app_state["kb_documents"] = 0

        return jsonify(result), 200

    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Delete error: {e}")
        return jsonify({"error": "Failed to delete data"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat with terrain analysis"""
    global rag_service

    try:
        if not request.json:
            return jsonify({"error": "Invalid request"}), 400

        question = request.json.get('message', '').strip()
        history = request.json.get('history', [])  # Accept conversation history

        if not question:
            return jsonify({"error": "No message provided"}), 400

        if models.llm is None:
            return jsonify({"error": "Chat service not available"}), 503

        app_state["total_queries"] += 1

        # Pass history to process_query - now returns 3 values
        response, mode, data_availability = rag_service.process_query(question, history)

        # Build response with data availability info
        result = {
            "success": True,
            "response": response,
            "mode": mode,
            "data_availability": data_availability
        }

        # Include terrain summary if coordinates were detected
        if rag_service.last_terrain_summary:
            result["terrain_summary"] = rag_service.last_terrain_summary

        return jsonify(result), 200

    except Exception as e:
        app_state["errors"] += 1
        app_state["last_error"] = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/chat",
            "error": str(e)
        }
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process message"}), 500


@app.route('/analyze_coordinates', methods=['POST'])
def analyze_coordinates():
    """
    Coordinate-based tactical analysis - now routes to RAGService

    This endpoint is maintained for backwards compatibility.
    RAGService now handles coordinate detection, terrain fetching,
    and tactical analysis with doctrine integration.
    """
    global rag_service

    try:
        if rag_service is None:
            return jsonify({"error": "RAG service unavailable"}), 503

        if not request.json:
            return jsonify({"error": "Invalid request"}), 400

        user_prompt = request.json.get('message', '').strip()
        # scenario parameter is now handled by _detect_scenario_type in RAGService

        if not user_prompt:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Coordinate-based tactical analysis (via RAGService): {user_prompt[:100]}...")
        app_state["total_queries"] += 1

        response, method, data_availability = rag_service.process_query(user_prompt)

        # Build response - include terrain summary if coordinates were detected
        result = {
            "success": True,
            "response": response,
            "strategy": response,  # Alias for backwards compatibility
            "method": method,
            "data_availability": data_availability,
            "models_used": ["Coordinate Parser", "OpenStreetMap API", "Open-Meteo Elevation API", Config.LLM_MODEL]
        }

        # Include terrain data if available
        if rag_service.last_terrain_summary:
            result["terrain_summary"] = rag_service.last_terrain_summary
            result["coordinates"] = rag_service.last_terrain_summary.get('coordinates')

        # Include raw terrain data for frontend display fields
        # (terrain_analysis, place_name, address, location, elevation, weather)
        if rag_service.last_terrain_data:
            td = rag_service.last_terrain_data
            result["terrain_data"] = {
                "terrain_analysis": td.get("terrain_analysis", {}),
                "place_name": td.get("place_name"),
                "address": td.get("address", {}),
                "location": td.get("location", {}),
                "elevation": td.get("elevation"),
                "weather": td.get("weather", {}).get("weekly_summary", {}),
            }

        return jsonify(result), 200

    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Coordinate analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to analyze coordinates"}), 500


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
        return jsonify({"error": "Debug query failed"}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("STARTING CHATBOT BACKEND (Coordinate-based)")
    logger.info("="*60)
    logger.info(f"Debug: {Config.DEBUG}")
    logger.info(f"Models: Ollama LLM({models.llm is not None}), Embeddings({models.embeddings is not None})")

    initialize_services()

    logger.info("="*60)
    logger.info(f"Starting Flask server on port {Config.PORT}")
    logger.info("="*60)

    app.run(debug=Config.DEBUG, port=Config.PORT, threaded=True)