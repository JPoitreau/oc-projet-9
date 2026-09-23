from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Optional, Literal
from app.chat import build_prompt_session, generate_response

from utils.utils import (
    get_embeddings, # noqa: F401
    search_relevant_segments, # noqa: F401
    load_index_and_raw, 
    PROJECT_ROOT, # noqa: F401
    RAW_DATA_PATH, # noqa: F401
    DATA_DIR, # noqa: F401
)

from app.db_creation import (
    rebuild_faiss_index, # noqa: F401
    check_for_dry_emails_urls_or_numeric, # noqa: F401
    normalize_text, # noqa: F401
    clean_text, # noqa: F401
    split_text_into_chunks # noqa: F401
)

# To run this API localy, navigate to the src directory and use the command:
# uvicorn app.api:app --reload

class MessageItem(BaseModel):                   #data model is a class that inherits from BaseModel, which is a Pydantic class
    role : Literal ["human", "ai", "system"]
    content: str

class UserRequest(BaseModel):
    user_question: str
    messages: Optional[List[MessageItem]] = None

chat_history = []

app = FastAPI()

load_dotenv()

INDEX_PATH = DATA_DIR / "faiss_index.idx"
METADATA_PATH = DATA_DIR / "metadata.csv"
if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    index = None
    raw = None
    print("Index and raw data have not been loaded. Please rebuild the index.")
else:
    index, raw = load_index_and_raw()

@app.post("/ask/")
def ask(request: UserRequest):
    """
    Args: request: UserRequest - a request from the user containing the user's question and the chat history
    Returns: response: str
    """
    global index, raw
    if index is None or raw is None:
        raise ValueError("Index and raw data have not been loaded. Please rebuild the index first.")

    if request.messages == [MessageItem(role='human', content='string')]:
        request.messages = []
        initial_message = "Comment puis-je vous aider?"
        request.messages.append(
            MessageItem(role = "ai", content=initial_message)
            )
    print("request.messages : ", request.messages)
    
    history_as_langchain = []

    for message_item in request.messages:
        if message_item.role == 'human':
            history_as_langchain.append(HumanMessage(message_item.content))
        elif message_item.role == 'ai':
            history_as_langchain.append(AIMessage(message_item.content))
        else:
            history_as_langchain.append(SystemMessage(message_item.content))

    print("history_as_langchain : ", history_as_langchain)

    prompt = build_prompt_session(history_as_langchain, raw, index, request.user_question)
    prompt.append(HumanMessage(request.user_question))
    print("prompt : ", prompt)

    request.messages = [
    MessageItem(role=msg.type, content=msg.content)
    for msg in prompt
    ]

    response = generate_response(prompt)

    print("response : ", response)
    
    request.messages.append(MessageItem(role="ai", content=response))
    return response

@app.get("/rebuild/")
def rebuild():
    global index, raw
    rebuild_result = rebuild_faiss_index()
    index, raw = load_index_and_raw()
    return rebuild_result






'''
@app.get("/predict/{employee_ID}") 
def predict_employee_departure(employee_ID: conint(ge = 0, le = 1469)):
    """
    Args: employee_ID : int compris entre 0 et 1469

    conint est une fonction qui contraint les données d'entrée à des integers 
    compris entre ge (greater or equal) et le (less or equal).
    Gère les erreurs de manière intrinsèque, sans avoir à ajouter soit même 
    les exceptions et messages d'erreur.
    """

    if (LOCAL).strip().lower() == "true":
        employee_ID = str(employee_ID)
        return predict_for_employee_local(
                                        employee_ID,
                                        DB_PATH, 
                                        MODEL_PATH
                                        )
    else:
        return predict_for_employee_deployed(
                            employee_ID,
                            DB_PATH,
                            MODEL_PATH
                            )'''