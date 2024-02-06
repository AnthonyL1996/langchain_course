from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

# Retrieval QA chain, retrieve some data from the DB to use as input for the users question

retriever = db.as_retriever() #

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"  # means stuffing them into the chain, as in injecting
)

# Chain types:
# map_reduce
# runs the language model multiple times
# For each interesting fact from the vector store, it will send a system message and human message prompt to the LLM.
# This can cause unrelated embeddings to be processed in the request.
# Uses the different summaries to generate a final answer.

# map_rerank
# Gives a score for each of the summarized prompts
# The score is based on the relevancy to the prompt inserted

# refine
# Builds an initial response and gives the LLM the opportunity to update it further with the context
# Runs multiple prompts but in series to use the information from the previous step.

# stuff
# is the fastest and gives you the best result most of the time

# Avoid duplicates because of the embedding created by the chain type stuff

result = chain.run("What is an interesting fact about the longest English language?")

print(result)
