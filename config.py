import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env once
load_dotenv()

# ---- Global config ----
DB_PATH = "rag_index.sqlite"
DOCS_DIR = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# ---- API keys / tokens ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# ---- Logging ----
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("simple_rag_bot")
