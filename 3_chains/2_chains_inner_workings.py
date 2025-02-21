from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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

# create individual runnables (steps in the chain)
# python lambda functions used here can also be replaced by regular functions.
# basically a function that takes an input, possibly does something with it and returns an output
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create the RunnableSequence (equivalent to the LCEL chain)
# the middle is an array that contains all the tasks between the first and last, in the proper order, regardless of their number
chain = RunnableSequence(first=format_prompt,middle=[invoke_model], last=parse_output)


# Run the chain
result = chain.invoke({"animal": "dogs", "fact_count": 3})
print(result)