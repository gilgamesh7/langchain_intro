import os

from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {lineno} - {message}", style='{')
logger = logging.getLogger("langchain_intro")

load_dotenv()

OPEN_API_KEY = os.environ.get('OPEN_API_KEY')


def main():
    logger.info("Starting langchain Intro")

if __name__ == "__main__":
    main()
