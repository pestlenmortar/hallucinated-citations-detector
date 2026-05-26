import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_PATH = os.getenv("DB_PATH", "papers.db")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("true", "1", "yes")
USE_LIVE_LOOKUP = os.getenv("USE_LIVE_LOOKUP", "true").lower() in ("true", "1", "yes")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "14"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "50"))
S2_RATE_LIMIT = float(os.getenv("S2_RATE_LIMIT", "1.0"))
