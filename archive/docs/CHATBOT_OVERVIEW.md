# Tactical Terrain Analysis Chatbot

A terrain-aware chatbot that performs Intelligence Preparation of the Battlefield (IPB) analysis by combining real geographic data, uploaded doctrine documents, and a locally-run LLM.

---

## Core Components

| Module | Role |
|---|---|
| `app.py` | Flask REST API server (port 5001). Exposes all endpoints and manages application state. |
| `services.py` | Business logic: **DocumentService** (upload, OCR, chunking, indexing) and **RAGService** (query routing, terrain analysis, doctrine retrieval, prompt building). |
| `models.py` | Loads and manages the LLM (Ollama), embedding model (HuggingFace), and vector store (Chroma). |
| `config.py` | Centralised configuration: model names, chunk sizes, folder paths, debug flags. |
| `utils.py` | Helpers for OCR preprocessing, text cleaning, smart chunking, and hybrid search. |
| `coordinate_parser.py` | Extracts coordinates from natural language. Supports decimal, labelled, DMS, MGRS, and UTM formats. |
| `terrain_data_fetcher.py` | Fetches real terrain data from free APIs (Open-Meteo, Nominatim, Overpass) and computes tactical metrics. |

## Key Features

**Coordinate Extraction** -- Parses coordinates in multiple formats (decimal, DMS, MGRS, UTM) directly from user messages.

**Real Terrain Intelligence** -- For any extracted coordinate, fetches elevation profiles, slope/gradient analysis, line-of-sight in eight directions, road networks, waterways, buildings, forests, crossings, power lines, cell towers, sensitive sites (schools, hospitals), helipads, and fuel stations. Calculates movement times for dismounted infantry, wheeled vehicles, APCs, and main battle tanks.

**Scenario-Aware Analysis** -- Detects mission type from the query (defensive, offensive, stability, reconnaissance, general) and tailors the IPB output accordingly using the OAKOC framework (Observation & Fields of Fire, Avenues of Approach, Key Terrain, Obstacles, Cover & Concealment) plus ASCOPE civil considerations.

**Document Upload & OCR** -- Accepts PDFs and images. Scanned documents are processed with Tesseract OCR (with image preprocessing). Text is split into overlapping chunks at three size levels (small, medium, large) and indexed in the vector store.

**Doctrine Retrieval** -- Hybrid search combining vector similarity with keyword matching. Terrain keywords are injected into queries to surface relevant doctrine passages. Sources are tracked for citation.

**Multi-Turn Conversations** -- Maintains conversation history and caches terrain data so follow-up questions reuse the same geographic context without re-fetching.

## AI Stack

- **LLM**: Qwen 3 8B (4-bit quantised) running locally via Ollama
- **Embeddings**: BAAI/bge-large-en-v1.5 (HuggingFace / Sentence Transformers)
- **Vector Store**: Chroma (persistent, on-disk)
- **External APIs** (free, no keys required): Open-Meteo (elevation), Nominatim (geocoding), Overpass (OpenStreetMap features)

## Conversation Flow

```
User message
  |
  v
1. Extract coordinates (if present)
  |
  v
2. Fetch terrain data from external APIs (if coordinates found)
   - Elevation, slope, LOS, infrastructure, civil features, movement times
  |
  v
3. Route the query:
   A. Coordinates found  --> full IPB analysis
   B. Follow-up question --> reuse cached terrain data
   C. No coordinates     --> prompt user to provide them
  |
  v
4. Detect scenario type (defensive / offensive / stability / recon / general)
  |
  v
5. Retrieve relevant doctrine via hybrid search
  |
  v
6. Build tactical prompt (rules, history, scenario guidance, doctrine, terrain, task)
  |
  v
7. LLM generates analysis --> return response with metadata
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (Ollama, embeddings, vector store status) |
| `/chat` | POST | Main chat endpoint; accepts `message` and `history` |
| `/upload` | POST | Upload a document (PDF/image) for analysis |
| `/upload_doctrine` | POST | Upload doctrine documents to the knowledge base |
| `/analyze_coordinates` | POST | Standalone coordinate-based tactical analysis |
| `/delete_all` | POST | Clear all documents and reset the vector store |
