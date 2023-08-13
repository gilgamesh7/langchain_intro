import os

from dotenv import load_dotenv
import logging

from langchain.llms import OpenAI # Pre 3.5 models
from langchain.chat_models import ChatOpenAI # modern models
from langchain.prompts.chat import(
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')
PROMPT_COUNTRY_INFO = """
    Provide information about {country}.
"""
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

        '''
            Simple - pre 3.5 models
        '''
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

        '''
            Simple 3.5 + Models
        '''
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


        '''
            Using Prompts
        '''
        country = input("Enter country name : ")
        message = HumanMessagePromptTemplate.from_template(template=PROMPT_COUNTRY_INFO)
        chat_prompt = ChatPromptTemplate.from_messages(messages=[message])
        chat_prompt_with_values = chat_prompt.format_prompt(country=country)
        response = llm(chat_prompt_with_values.to_messages())

        logger.info(f"Human prompt + Chat Prompt Template : {response}")

    except Exception as error:
        logger.error(f"{type(error).__name__} - {error}")   
        raise error 

if __name__ == "__main__":
    main()
