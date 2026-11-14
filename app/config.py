from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis Connection
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Embedding Model Setup
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384             # must match the model output
    
    # Semantic Cache Rules
    SEMANTIC_CACHE_THRESHOLD: float = 0.90  # tuned for your specific test pairs
    CACHE_TTL_SECONDS: int = 300            # 5 minutes before cached answers expire
    
    # Retrieval-Augmented Generation (RAG) Settings
    RRF_K: int = 60                         # constant used for Hybrid Search scoring
    TOP_K_RETRIEVAL: int = 3                # how many documents to fetch per query
    
    # Redis Index and Prefix Names for Documents
    DOC_INDEX: str = "idx:docs"
    DOC_PREFIX: str = "doc:"
    
    # Redis Index and Prefix Names for the Cache
    CACHE_INDEX: str = "idx:cache"
    CACHE_PREFIX: str = "cache:"
    EXACT_PREFIX: str = "exact:"

# Create a single shared instance of the settings
settings = Settings()