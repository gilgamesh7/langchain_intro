import os

from dotenv import load_dotenv
import logging

from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def setup()-> str:
    try:
        logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {lineno} - {message}", style='{')
        logger = logging.getLogger("langchain_intro")

        return logger
    except Exception as error:
        logger.error(f"{type(error).__name__} - {error}")   
        raise error 


def main():
    try:
        logger = setup()
        logger.info("Starting langchain Intro")

        logger.info("Create llm model")
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)

        logger.info("Q1 : Give me 5 facts about Rajinikanth")
        result = llm.predict(
            "Give me 5 facts about Rajinikanth"
        )
        logger.info(result)
    except Exception as error:
        logger.error(f"{type(error).__name__} - {error}")   
        raise error 

if __name__ == "__main__":
    main()
