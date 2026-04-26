# Potential Improvements

## Document Chunking

- Documents are split into 3 different sizes (small, medium, large), but the system doesn't actually use them differently — all sizes are searched the same way. Some further implementation is required to perfect the multi-resolution chunking strategy.

## Search Quality

- The system always retrieves exactly 10 results, regardless of how many are actually relevant. Low-quality matches can end up in the prompt and confuse the AI.

## External API Reliability

- If the Nominatim (place names) or Overpass (map features) APIs go down, the terrain analysis fails with no backup.

## Context Sent to the AI

- All retrieved document passages are joined together and sent to the AI as-is. For large retrievals, this can use up the AI's context window or bury the important information. Summarising or trimming low-relevance passages before sending would help.
