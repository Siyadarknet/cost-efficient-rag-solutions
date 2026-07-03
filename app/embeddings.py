from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingModel:
    def __init__(self):
        self.model_name = settings.EMBED_MODEL
        self.dimension = settings.EMBED_DIM
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Create embedding for a single text.
        """
        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple texts.
        """
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return vectors.tolist()


embedding_model = EmbeddingModel()