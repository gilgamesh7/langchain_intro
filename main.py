import os

from dotenv import load_dotenv
import logging

from langchain.llms import OpenAI # Pre 3.5 models
from langchain.chat_models import ChatOpenAI # modern models

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')

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

        logger.info("Create llm model for older GPT models , pre < 3.5")
        older_llm = OpenAI(openai_api_key=OPENAI_API_KEY)

        logger.info("Using pre 3.5 : Give me 5 facts about Rajinikanth")
        result = older_llm.predict(
            "Give me 5 facts about Rajinikanth"
        )
        logger.info(f"Prediction from pre 3.5 models : {result}")

        logger.info("Using pre 3.5 : Who is the Prime Minister of New Zealand")
        result = older_llm.predict(
            "Who is the Prime Minister of New Zealand"
        )
        logger.info(f"Prediction from pre 3.5 models : {result}")

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
        logger.info(f"Using {OPENAI_MODEL} model : Give me 5 facts about Rajinikanth")
        result = llm.predict(
            "Give me 5 facts about Rajinikanth"
        )
        logger.info(f"Prediction from {OPENAI_MODEL} model : {result}")

        logger.info(f"Using {OPENAI_MODEL} model : Who is the Prime Minister of New Zealand")
        result = llm.predict(
            "Who is the Prime Minister of New Zealand"
        )
        logger.info(f"Prediction from {OPENAI_MODEL} model : {result}")


    except Exception as error:
        logger.error(f"{type(error).__name__} - {error}")   
        raise error 

if __name__ == "__main__":
    main()
