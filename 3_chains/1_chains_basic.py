from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_count} facts."),
]

# Define prompt templates (no need for seperate runnable chains)
prompt_template = ChatPromptTemplate.from_messages(messages)


# create the combined chain using LangChain Expression Language (LCEL)
# StrOutputParser() extracts the value of "content" property from the entire response object
# The pipe operator here passes the result from one task in the chain to the next task
chain = prompt_template | llm | StrOutputParser()
# chain = prompt_template | llm

# Run the chain
# the variables passed to the invoke method will be avaiable to all prompt templates in the chain, if they are more than one
result = chain.invoke({"animal": "cats", "fact_count": 1})
print(result)