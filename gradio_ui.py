import gradio as gr

from config import logger
from rag_core import index_documents, rag_answer


def launch_gradio_ui():
    """
    Launch a local Gradio UI to debug the RAG pipeline.
    """
    logger.info("Indexing documents for Gradio UI...")
    index_documents()

    iface = gr.Interface(
        fn=rag_answer,
        inputs=gr.Textbox(
            lines=2,
            label="Your question",
            placeholder="Ask something about your docs..."
        ),
        outputs=[
            gr.Textbox(label="Answer", lines=5),
            gr.Textbox(label="Context & sources (for debugging)", lines=15),
        ],
        title="Local RAG Debugger",
        description=(
            "Ask questions over the same docs and pipeline used by the Telegram bot.\n"
            "Useful for local testing and debugging."
        ),
    )

    iface.launch()



