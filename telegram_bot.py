from collections import defaultdict, deque

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from config import TELEGRAM_BOT_TOKEN, logger
from rag_core import (
    index_documents,
    run_rag_pipeline,
    get_help_text,
)

# Per-user history (last 3 Q&A)
user_history: dict[int, deque[tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=3)
)


# Handlers
async def start_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_help_text())


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    query = " ".join(context.args).strip()
    await update.message.reply_text("Thinking… retrieving relevant info from docs...")

    answer, sources_lines = run_rag_pipeline(query)

    reply = f"Answer:\n{answer}"
    if sources_lines:
        reply += "\n\nSources used:\n" + "\n".join(sources_lines)

    await update.message.reply_text(reply)

    user_id = update.effective_user.id
    user_history[user_id].append((query, answer))


async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "This bot is running the Mini-RAG (text) variant.\n"
        "Image mode is not enabled; I only handle text questions over the docs."
    )
    await update.message.reply_text(text)


async def summarize_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = user_history.get(user_id)

    if not history:
        await update.message.reply_text(
            "I don't have any recent interactions to summarize yet.\n"
            "Ask me something first using /ask, then try /summarize."
        )
        return

    convo_text = ""
    for i, (q, a) in enumerate(history, start=1):
        convo_text += f"Q{i}: {q}\nA{i}: {a}\n\n"

    summary_question = (
        "Please provide a short, clear summary (2–3 sentences) of the conversation above."
    )

    from rag_core import call_llm  # local import to avoid circular issues if any
    try:
        summary = call_llm(convo_text, summary_question)
    except Exception as e:
        logger.exception("Error calling LLM for /summarize: %s", e)
        await update.message.reply_text("Error summarizing the conversation.")
        return

    await update.message.reply_text(f"Summary of your recent interactions:\n{summary}")


async def on_startup(app):
    logger.info("Indexing documents...")
    index_documents()


def run_telegram_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN environment variable.")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    application.add_handler(CommandHandler("help", start_help))
    application.add_handler(CommandHandler("start", start_help))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(CommandHandler("image", image_cmd))
    application.add_handler(CommandHandler("summarize", summarize_cmd))

    logger.info("Bot starting...")
    application.run_polling()

