"""
Text embedding model.

Converts transaction descriptions to vectors so we can compare them mathematically.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps a pre-trained AI model for converting text to embeddings.
    
    Uses all-MiniLM-L6-v2: fast, good quality, works on plain text descriptions.
    """

    def __init__(self):
        """Load the embedding model."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        """
        Convert list of text descriptions to embedding vectors.
        
        Input: ["GROCERY STORE", "GAS STATION", ...]
        Output: [[vector1], [vector2], ...] - each vector is 384 dimensions
        """
        return self.model.encode(texts, show_progress_bar=False)