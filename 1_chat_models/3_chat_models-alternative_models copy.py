from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage("You are a social media content strategist"), #defines the AIs role
    HumanMessage("Give a short tip to create engaging post on Instagram"),
]

result = llm.invoke(messages)

print(result.content)