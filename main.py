import os

from dotenv import load_dotenv
import logging

from langchain.llms import OpenAI # Pre 3.5 models
from langchain.chat_models import ChatOpenAI # modern models
from langchain.prompts.chat import(
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain

class Country(BaseModel):
    capital : str = Field(description='Capital of the country')
    name: str = Field(description='Name of the country')

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')
PROMPT_COUNTRY_INFO = """
    Provide information about {country}.
"""
PROMPT_COUNTRY_INFO_WITH_INSTRUCTIONS = """
    Provide information about {country}.
    {format_instructions}.
    If the country does not exist make up something.
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

        '''
            Return Pydantic response
        '''
        parser = PydanticOutputParser(pydantic_object=Country)
        message = HumanMessagePromptTemplate.from_template(template=PROMPT_COUNTRY_INFO_WITH_INSTRUCTIONS)
        chat_prompt = ChatPromptTemplate.from_messages(messages=[message])
        chat_prompt_with_values = chat_prompt.format_prompt(country=country, format_instructions=parser.get_format_instructions())
        response = llm(chat_prompt_with_values.to_messages())
        data = parser.parse(response.content)

        logger.info(f"Human prompt + Chat Prompt Template + Pydantic class: {data}")
        logger.info(f"Human prompt + Chat Prompt Template + Pydantic class: {data.capital}")

        '''
            Get llm to call an API for you
        '''
        chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=False)
        result = chain_new.run("What is the weather like right now in Auckland New Zealand in degrees Celcius")
        logger.info(f"API Chain weather for AKL : {result}")


    except Exception as error:
        logger.error(f"{type(error).__name__} - {error}")   
        raise error 

if __name__ == "__main__":
    main()
