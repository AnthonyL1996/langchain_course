from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma  # Wrapped up version of Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
# emb = embeddings.embed_query("hi there")
# print(emb)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # 200 characters, finds 200 characters and then uses the seperator
    chunk_overlap=0,  # Overlap between chunks to not have awkward chunks with missing parts

)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# Already reaches out to OpenAI to calculate the embeddings
# Every time you run this you will add duplicate data
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)



results = db.similarity_search_with_score(
    "What is an interesting fact about the english language?",
    k=2  # Amount of relevant answers you want
)

for result in results:  # Tuple with document content and score
    print("\n")
    print(result[1])  # Search score
    print(result[0].page_content)  # Document content

# How to find most relevant text in the facts? Scan the facts for the words that match.
# Not that reliable, there is no interpretation in the search
# The alternative are embeddings for scemantic search

# Model to generate embeddings, you can run this locally or from an API such as from OpenAI
# Embeddings are not compatible they need to come from the same algorithm/model
