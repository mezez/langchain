# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project and FireStore Database
3. Retrieve the Project ID
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. pip install langchain-google-firestore
6. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

# create a ChatOpenAI model
llmModel = ChatOpenAI(model="gpt-3.5-turbo")

# setup Firebase firestore for chat history
PROJECT_ID = "langchain-3c8e2"
SESSION_ID = "user_session_new"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

chat_history = None
firestore_client = None

def initialise_firestore_client():
    global firestore_client
    print("Initialising firestore client")
    firestore_client = firestore.Client(project=PROJECT_ID)

def initialise_firestore_chat_history():
    global chat_history
    print("Initialising firestore chat history")
    chat_history = FirestoreChatMessageHistory(session_id=SESSION_ID, collection=COLLECTION_NAME, client=firestore_client)
    print("Chat history initialised")
    print("Current chat history: ", chat_history.messages)



def user_wants_to_quit_chat(user_message):
    if user_message.lower() == "exit":
        return True
    return False

def get_AI_response_with_message_history():
    global chat_history
    result = llmModel.invoke(chat_history.messages)
    response = result.content
    return response


def chat_loop():
    while True:
        query = input("You: ")
        if user_wants_to_quit_chat(query):
            break
        
        chat_history.add_user_message(query)
        AI_response = get_AI_response_with_message_history()
        chat_history.add_ai_message(AI_response)

        print(f"AI: {AI_response}")

def show_message_history():
    print("---- Message History ----")
    print(chat_history.messages)


initialise_firestore_client()

initialise_firestore_chat_history()

# start chat
chat_loop()

# show message history at the end
show_message_history()
