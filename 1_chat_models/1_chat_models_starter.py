from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# llm = ChatOpenAI(model="gpt-4o") # latest models are usually the most expensive. Maybe try GPT 3.5, 4, etc
llm = ChatOpenAI(model="gpt-3.5-turbo") 
# llm = ChatOpenAI(model="gpt-3.5") 

# call to llm
result = llm.invoke("What is the square root of 49")

print(result.content)
