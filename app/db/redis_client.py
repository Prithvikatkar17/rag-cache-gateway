import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import ResponseError
from app.config import settings

class RedisManager:
    def __init__(self):
        self.client: redis.Redis | None = None

    async def connect(self):
        # decode_responses=False so the raw float32 vector bytes survive (Day 4)
        self.client = redis.Redis(
            host=settings.REDIS_HOST, 
            port=settings.REDIS_PORT,
            decode_responses=False
        )
        await self.client.ping()
        
        # Set up the index for our RAG documents
        await self._ensure_index(settings.DOC_INDEX, settings.DOC_PREFIX, "content")
        
        # Set up the index for our saved Q&A cache
        await self._ensure_index(settings.CACHE_INDEX, settings.CACHE_PREFIX, "prompt")

    def _vector_field(self):
        # Centralized vector setup using the dimension from config.py
        return VectorField("embedding", "HNSW",
                           {"TYPE": "FLOAT32", "DIM": settings.VECTOR_DIMENSION,
                            "DISTANCE_METRIC": "COSINE"})

    async def _ensure_index(self, name: str, prefix: str, text_field: str):
        # TODO 1: Check if the index already exists before trying to create it
        try:
            await self.client.ft(name).info()
            return  # Index exists! We can safely exit this function.
        except ResponseError:
            pass    # Index doesn't exist yet, so we catch the error and keep going.

        # TODO 2: Build the schema and create the index
        schema = (
            TextField(text_field), 
            self._vector_field()
        )
        await self.client.ft(name).create_index(
            schema,
            definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )

    async def close(self):
        if self.client:
            await self.client.aclose()

# Create a single shared instance to be imported across the app
redis_manager = RedisManager()