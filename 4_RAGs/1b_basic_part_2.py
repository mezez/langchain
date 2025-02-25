import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

#define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # the model must match the same used for creating the embeddings

# load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# define the user's question
query = "Where does Gandalf meet Frodo?"

# retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k":3, "score_threshold": 0.5}) # return the top 3 chunks with the highest relevance scores, with similarity scores >= 0.5 (scale of 0 - 1)
relevant_docs = retriever.invoke(query)

# display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

#These relevant docs can now be given to a model to answer the question/query, eg see 2b
