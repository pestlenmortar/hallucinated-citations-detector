import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_PATH = os.getenv("DB_PATH", "papers.db")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("true", "1", "yes")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
