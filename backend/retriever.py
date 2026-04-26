import logging
from utils import preprocess_query, similarity_search, resolve_parents

logger = logging.getLogger(__name__)


class DoctrineRetriever:
    """Search the doctrine vector store and return relevant passages."""

    def __init__(self, vectorstore, parent_store):
        self.vectorstore = vectorstore
        self.parent_store = parent_store

    def retrieve(self, query: str, terrain_data: dict = None) -> str:
        """Return formatted doctrine passages relevant to query, or empty string."""
        if self.vectorstore is None:
            return ""

        try:
            enhanced_query, _ = preprocess_query(query)

            if terrain_data:
                enhanced_query = self._enhance_with_terrain(enhanced_query, terrain_data)

            child_docs = similarity_search(enhanced_query, self.vectorstore, k=5)
            if not child_docs:
                return ""

            docs = resolve_parents(child_docs, self.parent_store) if self.parent_store else child_docs

            logger.info(f"Doctrine: {len(child_docs)} child chunks → {len(docs)} context chunks")

            parts = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown").replace(".pdf", "").replace("_", " ")
                page = doc.metadata.get("page", "")
                label = f"[{source}, p.{page}]" if page else f"[{source}]"
                parts.append(f"{label}\n{doc.page_content}")

            return "\n\n".join(parts)

        except Exception as e:
            logger.warning(f"Doctrine retrieval failed: {e}")
            return ""

    def _enhance_with_terrain(self, query: str, terrain_data: dict) -> str:
        """Append terrain-derived keywords to improve doctrine retrieval relevance."""
        keywords = []

        slope_data = terrain_data.get("slope_analysis", {})
        los_data = terrain_data.get("line_of_sight", {})
        analysis = terrain_data.get("terrain_analysis", {})

        if analysis.get("high_ground") or los_data.get("is_high_ground"):
            keywords.append("high ground observation fields of fire")

        buildings = terrain_data.get("buildings", [])
        if analysis.get("urban_terrain") or len(buildings) > 50:
            keywords.append("urban operations MOUT complex terrain")

        cover = analysis.get("cover_availability", "")
        forests = terrain_data.get("forests", [])
        if cover == "excellent" or len(forests) > 10:
            keywords.append("cover and concealment")
        elif cover == "limited" or (len(buildings) < 10 and len(forests) < 5):
            keywords.append("open terrain exposed")

        if terrain_data.get("waterways"):
            keywords.append("water obstacle river crossing")

        if terrain_data.get("crossings"):
            keywords.append("bridge crossing point")

        mobility = slope_data.get("mobility", "")
        if mobility in ["restricted", "severely_restricted"]:
            keywords.append("restricted terrain mobility")

        if len(terrain_data.get("roads", [])) > 20:
            keywords.append("avenues of approach road network")

        if keywords:
            return f"{query} {' '.join(keywords)} OCOKA IPB terrain analysis"
        return query
