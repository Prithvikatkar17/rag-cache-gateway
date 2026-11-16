import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingModel:
    def __init__(self):
        # We start with nothing loaded to save memory until the server actually turns on.
        self.model: SentenceTransformer | None = None

    def load(self):
        # Loads the AI model using the exact name from your config.py
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
        # A quick safety check: we embed a dummy sentence to measure its size.
        size = len(self.model.encode("dimension check"))
        
        # If the model outputs 768 dimensions but config says 384, we crash immediately.
        # This prevents confusing Redis errors later.
        if size != settings.VECTOR_DIMENSION:
            raise ValueError(
                f"model gives {size} numbers but VECTOR_DIMENSION is {settings.VECTOR_DIMENSION}"
            )

    def embed(self, text: str) -> np.ndarray:
        """Blocking: one sentence in, one float32 vector out."""
        # TODO 1: Convert the sentence into numbers, and force them to be 32-bit floats
        return self.model.encode(text).astype(np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Blocking: a list of sentences in, a table of float32 vectors out."""
        return self.model.encode(texts).astype(np.float32)

    async def aembed(self, text: str) -> np.ndarray:
        """The version the API will use: runs embed() in a worker thread."""
        # TODO 2: Send the heavy AI math to a separate thread so the server doesn't freeze
        return await asyncio.to_thread(self.embed, text)

    @staticmethod
    def to_bytes(vec: np.ndarray) -> bytes:
        # Redis databases only accept raw computer bytes, not Python numpy arrays
        return vec.astype(np.float32).tobytes()

# We create exactly ONE shared instance of this model for the whole app.
embedding_model = EmbeddingModel()