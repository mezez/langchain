import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# define the directory containing the text file and the persistent directory
curent_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curent_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(curent_dir,  "db", "chroma_db")

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initialising vector store...")

    # ensure the text files exist
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    # read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50) # ideally between 20 - 100 is okay for chunk overlap
    docs = text_splitter.split_documents(documents)

    # display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialise")