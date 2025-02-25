import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# define the user's question
query = "Where is Dracula's castle located?"

# retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3, "score_threshold": 0.2}
)
relevant_docs = retriever.invoke(query)

# display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")


combined_input = (
    "here are some documents that might help anwser the question: " 
    + query 
    + "\n Relevant Documents \n" 
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the anwser is not found in the documents, respond with 'I'm not sure'."
)

# create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# define the message for the model
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

print(messages, "messages")

# invoke the model with the combined input
result = model.invoke(messages)

# display the full result and content only
print("\n--- Generated Response ---")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
