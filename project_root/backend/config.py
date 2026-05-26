import os

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)

DB_PATH = _resolve(os.getenv("DB_PATH", "papers.db"))
FAISS_INDEX_PATH = _resolve(os.getenv("FAISS_INDEX_PATH", "faiss_index.bin"))
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("true", "1", "yes")
USE_LIVE_LOOKUP = os.getenv("USE_LIVE_LOOKUP", "true").lower() in ("true", "1", "yes")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
