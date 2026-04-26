import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from models import get_models
from document_service import DocumentService
from retriever import DoctrineRetriever
from terrain_data_fetcher import TerrainDataFetcher
from utils import ParentChunkStore
from agent import TacticalAgent
import tools.terrain as terrain_tool
import tools.doctrine as doctrine_tool
import tools.military as military_tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

app_state = {
    "started_at": datetime.now().isoformat(),
    "documents_processed": 0,
    "kb_documents": 0,
    "total_queries": 0,
    "errors": 0,
    "last_error": None,
}

models = get_models()
document_service = None
agent = None


def initialize_services():
    global document_service, agent

    try:
        document_service = DocumentService(models.vectorstore, models.parent_store)

        # Wire up tools with their dependencies.
        terrain_tool.initialize(TerrainDataFetcher())

        retriever = DoctrineRetriever(models.vectorstore, models.parent_store)
        doctrine_tool.initialize(retriever)

        db_path = Path(__file__).parent.parent / "data" / "military" / "military_power_prompt.json"
        military_tool.initialize(db_path)

        agent = TacticalAgent(models.llm)

        logger.info("All services initialized")

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        logger.error(traceback.format_exc())


@app.route("/health", methods=["GET"])
def health():
    try:
        ollama_ok, ollama_msg = models.check_ollama_connection()
        embed_ok, embed_msg = models.check_embeddings()
        is_healthy = ollama_ok and embed_ok

        return jsonify({
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ollama": {"status": "ok" if ollama_ok else "error", "message": ollama_msg},
                "embeddings": {"status": "ok" if embed_ok else "error", "message": embed_msg},
                "vector_store": {"status": "ok" if models.vectorstore is not None else "empty"},
            },
            "stats": {
                "documents_processed": app_state["documents_processed"],
                "kb_documents": app_state["kb_documents"],
                "total_queries": app_state["total_queries"],
                "errors": app_state["errors"],
            },
        }), 200 if is_healthy else 503

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": "Health check failed"}), 500


@app.route("/upload", methods=["POST"])
def upload_document():
    global document_service
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        allowed = [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        if not any(file.filename.lower().endswith(ext) for ext in allowed):
            return jsonify({"error": "Only PDF and image files supported"}), 400

        if models.llm is None or models.embeddings is None:
            return jsonify({"error": "Server components not initialized"}), 500

        safe_filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        result = document_service.upload_and_index(filepath, file.filename)
        app_state["documents_processed"] += 1

        return jsonify({"success": True, "message": f"Successfully processed {file.filename}", "details": result}), 200

    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process uploaded file"}), 500


@app.route("/upload_doctrine", methods=["POST"])
def upload_doctrine():
    global document_service
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        safe_filename = secure_filename(file.filename)
        kb_filename = f"KB_{safe_filename}"
        filepath = os.path.join(Config.KB_FOLDER, kb_filename)
        file.save(filepath)

        result = document_service.upload_and_index(filepath, kb_filename, is_kb=True)
        app_state["kb_documents"] += 1

        return jsonify({
            "success": True,
            "filename": kb_filename,
            "chunks": result["chunks"],
            "text_length": result["text_length"],
            "file_size_kb": result["file_size_kb"],
        }), 200

    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Doctrine upload error: {e}")
        return jsonify({"error": "Failed to process doctrine document"}), 500


@app.route("/delete_all", methods=["POST"])
def delete_all():
    global document_service
    try:
        if not request.json or not request.json.get("confirm"):
            return jsonify({"error": 'Must send {"confirm": true} to delete all data'}), 400

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


@app.route("/chat", methods=["POST"])
def chat():
    global agent
    try:
        if not request.json:
            return jsonify({"error": "Invalid request"}), 400

        question = request.json.get("message", "").strip()
        history = request.json.get("history", [])

        if not question:
            return jsonify({"error": "No message provided"}), 400

        if models.llm is None:
            return jsonify({"error": "Chat service not available"}), 503

        app_state["total_queries"] += 1

        response, mode, data_availability = agent.run(question, history)

        result = {
            "success": True,
            "response": response,
            "mode": mode,
            "data_availability": data_availability,
        }

        if agent.last_terrain_summary:
            result["terrain_summary"] = agent.last_terrain_summary

        return jsonify(result), 200

    except Exception as e:
        app_state["errors"] += 1
        app_state["last_error"] = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/chat",
            "error": str(e),
        }
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process message"}), 500


@app.route("/analyze_coordinates", methods=["POST"])
def analyze_coordinates():
    """Backwards-compatible endpoint — routes to the agent."""
    global agent
    try:
        if agent is None:
            return jsonify({"error": "Agent unavailable"}), 503
        if not request.json:
            return jsonify({"error": "Invalid request"}), 400

        user_prompt = request.json.get("message", "").strip()
        if not user_prompt:
            return jsonify({"error": "No message provided"}), 400

        app_state["total_queries"] += 1

        response, method, data_availability = agent.run(user_prompt)

        result = {
            "success": True,
            "response": response,
            "strategy": response,
            "method": method,
            "data_availability": data_availability,
            "models_used": ["CoordinateParser", "OpenStreetMap API", "Open-Meteo", Config.LLM_MODEL],
        }

        if agent.last_terrain_summary:
            result["terrain_summary"] = agent.last_terrain_summary
            result["coordinates"] = agent.last_terrain_summary.get("coordinates")

        if agent.last_terrain_data:
            td = agent.last_terrain_data
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


@app.route("/debug/chunks", methods=["GET"])
def debug_chunks():
    try:
        if models.vectorstore is None:
            return jsonify({"error": "No documents loaded", "chunks": []}), 404

        results = models.vectorstore.similarity_search("", k=20)
        chunks_info = [
            {
                "index": idx,
                "content_preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                "content_length": len(doc.page_content),
                "metadata": doc.metadata,
            }
            for idx, doc in enumerate(results)
        ]

        return jsonify({
            "total_chunks": len(results),
            "chunks": chunks_info,
            "raw_documents": len(document_service.raw_documents) if document_service else 0,
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


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING TACTICAL CHATBOT BACKEND")
    logger.info("=" * 60)
    logger.info(f"Debug: {Config.DEBUG}")
    logger.info(f"LLM: {Config.LLM_MODEL}")

    initialize_services()

    logger.info("=" * 60)
    logger.info(f"Starting Flask server on port {Config.PORT}")
    logger.info("=" * 60)

    app.run(debug=Config.DEBUG, port=Config.PORT, threaded=True)
