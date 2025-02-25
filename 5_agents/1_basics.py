from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime
from langchain.agents import tool

load_dotenv()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

llm = ChatOpenAI(model="gpt-3.5-turbo")

query = "What is the current time"

prompt_template = hub.pull("hwchase17/react") # SEE https://smith.langchain.com/hub/hwchase17/react

# set tools that can be used by the agent for carring out tasks or solving problems
tools = [get_system_time]

# create agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


result = agent_executor.invoke({"input": query})
