import sys
from config import logger
from telegram_bot import run_telegram_bot
from gradio_ui import launch_gradio_ui

def main():
    """
    Usage:
        python main.py           # default: telegram
        python main.py telegram  # telegram mode explicitly
        python main.py gradio    # gradio debug UI
    """
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "telegram"

    if mode == "telegram":
        logger.info("Running in TELEGRAM mode")
        run_telegram_bot()
    elif mode == "gradio":
        logger.info("Running in GRADIO mode")
        launch_gradio_ui()
    else:
        raise ValueError("Unknown mode. Use 'telegram' or 'gradio'.")


if __name__ == "__main__":
    main()
