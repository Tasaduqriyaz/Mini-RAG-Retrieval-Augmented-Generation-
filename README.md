# Avivo.ai Mini-RAG Bot (Telegram + Gradio)

This project is a **Mini-RAG (Retrieval-Augmented Generation)** system that:

- Indexes `.txt` and `.md` documents from a `docs/` directory  
- Stores chunk embeddings in SQLite  
- Retrieves relevant chunks for any user question  
- Uses OpenAI for final response generation  
- Works through **both Telegram Bot** and **Gradio UI**  
- Shows source snippets + similarity scores  
- Implements query embedding caching for efficiency  

---

# 1. Project Structure

```
project/
│
├── config.py
├── rag_core.py
├── telegram_bot.py
├── gradio_ui.py
├── main.py
└── Screenshots   # contains screenshots captured while running the code (telegram bot and gradio UI)
└── docs/


```

---

# 2. Module Overview

### **`config.py`**
- Loads `.env` variables  
- Configures logging  
- Stores constants:  
  - `DB_PATH`, `DOCS_DIR`, `TOP_K`, model name  
  - OpenAI + Telegram tokens  
- Ensures all modules access a shared configuration

---

### **`rag_core.py`**
The heart of the RAG system.

Implements:

- SQLite DB setup  
- Document chunking  
- Embedding using `sentence-transformers/all-MiniLM-L6-v2`  
- Query embedding caching  
- Cosine similarity retrieval  
- LLM call using OpenAI  
- Shared pipeline: `run_rag_pipeline()`  
- `rag_answer()` (special wrapper for Gradio)

---

### **`telegram_bot.py`**
Provides the Telegram interface using `python-telegram-bot`.

Implements commands:

- `/start` – welcome message  
- `/help` – help text  
- `/ask` – RAG query  
- `/summarize` – summarize last 3 conversations  
- `/image` – disabled placeholder  

Per-user memory implemented with:

```
user_history[user_id] = deque(maxlen=3)
```

---

### **`gradio_ui.py`**
Provides a browser interface with:

- Input textbox  
- Answer display  
- Sources/snippets display  
- Chat-like interaction history  
- Summarization of session interactions  

Runs through:

```
python main.py gradio
```

---

### **`main.py`**
Entry point.

Usage:

```
python main.py          # run Telegram bot
python main.py gradio   # run Gradio UI
```

---

# 3. Installation Using `uv`

This project is designed for **uv**, the modern Python package manager.

### Step 1 — Create the project

```
uv init
```

This creates a `pyproject.toml`.

---

### Step 2 — Add dependencies

Edit `pyproject.toml`:

```toml
[project]
name = "avivo-mini-rag-bot"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [ 
    "python-telegram-bot==20.8",
    "sentence-transformers",
    "numpy",
    "openai==0.28",
    "tqdm",
    "numpy",
    "sentence_transformers",
    "telegram",
    "hf-xet>=1.2.0",
    "python-dotenv>=1.2.1",
    "gradio>=6.0.2",
 
]
```

---

### Step 3 — Install everything

```
uv sync
```

Creates `.venv/` and installs all dependencies.

---

### Step 4 — Activate the environment

Windows PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```
source .venv/bin/activate
```

---

# 4. Add Your API Keys

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

---

# 5. How to Get a Telegram Bot Token (BotFather)

1. Open Telegram  
2. Search for **BotFather** (verified account)  
3. Start the chat and send:

```
/start
```

4. Create a bot:

```
/newbot
```

5. Provide:
   - Display name  
   - Username ending with `bot`  

6. BotFather returns:

```
Here is your bot token:
1234567890:ABCDEF1234567890abcdef1234567890
```

7. Copy this token into your `.env`:

```
TELEGRAM_BOT_TOKEN=1234567890:ABCDEF1234567890abcdef1234567890
```

---

# 6. Adding Documents

Create a `docs/` directory:

```
mkdir docs
```

Add `.txt` or `.md` files, for example:

- `company_policy.md`
- `privacy_guidelines.md`
- `internal_policy.md`
- `quick_start_guide.md`

These are indexed at startup.

---

# 7. Running the Project

### **Run Telegram Bot**

```
python main.py
```

Commands:

- `/start` – help  
- `/ask <question>` – RAG query  
- `/summarize` – summarizes last 3 interactions  
- `/image` – placeholder  

---

### **Run Gradio UI**

```
python main.py gradio
```

Opens a browser window with interactive UI.

---

# 8. How RAG Works

1. Documents chunked into max ~500 chars  
2. Embeddings generated using SentenceTransformers  
3. Stored in SQLite (`embedding` stored as JSON)  
4. Query → embedding → cosine similarity  
5. Top-k chunks used to build context  
6. OpenAI model (`gpt-4o-mini`) answers ONLY using context  
7. Sources shown (with scores + snippets)

---

# 9. Features Implemented for Evaluation

✔ Clean modular code  
✔ Small model footprint  
✔ Efficient caching  
✔ Telegram + Web UI  
✔ Summarization  
✔ Source snippets  
✔ Logging + environment handling  

---

# 10. Summary

This project delivers a fully functional Mini-RAG system with:

- Clean architecture  
- Multi-interface usage  
- Efficient retrieval  
- Helpful summarization  
- Minimal dependencies  

Perfect for assignments, demos, interview projects, or small production prototypes.

# 11. Models and APIs Used

This project is intentionally lightweight and optimized to meet evaluation criteria such as *efficiency, clarity, and minimal dependencies*.  
Below is an overview of all models and APIs used in the system.

## 1. Embedding Model – `sentence-transformers/all-MiniLM-L6-v2`
- Converts document chunks and user queries into vector embeddings  
- Fast CPU performance, low memory (~80MB)  
- Ideal for small-scale, low-latency RAG systems  

## 2. Language Model (LLM) – `gpt-4o-mini`
- Used for final answer generation and summarization  
- Low latency, cost-effective, strong grounding  

## 3. APIs & Libraries Used
| Component | Purpose |
|----------|---------|
| OpenAI API | LLM text generation |
| sentence-transformers | Embedding model |
| python-telegram-bot | Telegram interface |
| Gradio | Local UI |
| SQLite3 | Lightweight vector store |
| NumPy | Vector math |
| dotenv | Environment management |
| tqdm | Progress display |

---

# 12. System Architecture Diagram

```
                          ┌────────────────────────┐
                          │     docs/ directory    │
                          │ (.txt / .md documents) │
                          └─────────────┬──────────┘
                                        │
                                 Chunking Logic
                                        │
                          SentenceTransformer Embeddings
                                        │
                        ┌───────────────▼────────────────┐
                        │            SQLite DB            │
                        │   chunks (text) + embeddings    │
                        └───────────────┬────────────────┘
                                        │
                                   User Query
                                        │
                                 Query Embedding
                                        │
                                 Cosine Similarity
                                        │
                          ┌─────────────▼───────────────┐
                          │        Top-K Retriever       │
                          │   (most relevant chunks)     │
                          └─────────────┬───────────────┘
                                        │
                                RAG Context Builder
                                        │
                            OpenAI gpt-4o-mini Call
                                        │
                         ┌──────────────▼───────────────────┐
                         │    Final Contextual Answer       │
                         └──────────────┬───────────────────┘
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │                                                    │
      Telegram Bot (/ask, /summarize)                    Gradio UI (chat panel)
              │                                                    │
              └────────────────────────────────────────────────────┘
```