import logging 

from dotenv import dotenv_values
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import textbase
from textbase.message import Message
from textbase import models
from typing import List
from logs import log_to_termianal


config = dotenv_values(".env")
DB_FAISS_PATH = 'VectorStores/db_faiss'


# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT= """ Using the following pieces of information, answer the users query. If the question asked by the user is very general and doesn't relate to any conditions given in the context return an appropriate answer
don't try to makeup an answer. 

"""

@log_to_termianal
@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    logging.info("loading hugging face embeddings and vector store ")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'}) 
    db = FAISS.load_local(DB_FAISS_PATH,embeddings=embeddings)

    latest_call = message_history[-1] # retreiving the latest messeage from user 

    search = db.similarity_search(str(latest_call.content),k=5,fetch_k=20)
    res = search[0].lc_kwargs['page_content']
    res = f'''\n\ncontext: {res}

            Question: {latest_call.content}

            Return The Most helpful and relevant answer for the user based on the given context
            Helpful answer: '''
    logging.info(f"The message parameter is {message_history} ")
    
    
    models.OpenAI.api_key = config['OPENAI_API_KEY']
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT + res,
        message_history=message_history,
        model="gpt-3.5-turbo",
        temperature=0.5
    )

    return bot_response, state


if __name__ == '__main__':
    pass