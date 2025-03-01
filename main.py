import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

embeddings_model="text-embedding-3-small"
chat_model="gpt-4o-mini"
# # # List available models
# client = OpenAI()

# model_list=client.models.list()
# for model in model_list.data:
#     print(model.id)


print(f"initializing llm with model {chat_model}")
llm = ChatOpenAI( model=chat_model,)

print(f"initializing embeddings with model {embeddings_model}\n")
embeddings = OpenAIEmbeddings( model=embeddings_model,)
print("creating index")
num_dimensions = len(embeddings.embed_query("hello world"))
print(f"num_dimensions: {num_dimensions}\n")
index = faiss.IndexFlatL2(num_dimensions )

print("creating vector store\n")
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

print("loading pdf\n")
loader = PyPDFLoader("Terms_and_Conditions_C.pdf")
docs = loader.load()

print("splitting docs\n")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
# for doc in split_docs:
#     print(doc.metadata)
#     print(doc.page_content)
#     print("\n")
print("creating docs list\n")
docs_list = [Document( page_content=doc.page_content, metadata=doc.metadata,) for doc in split_docs]
    
uuids = [str(uuid4()) for _ in range(len(docs_list))]

print("adding documents to vector store\n")
vector_store.add_documents(documents=docs_list, ids=uuids)

retriever = vector_store.as_retriever()
############################################################
# Define the retrieval tool
def retrieve_docs(query):
    return retriever.get_relevant_documents(query,k=5)

print("defining retrieval tool\n")
retrieval_tool = Tool(
    name="Document Retriever",
    func=retrieve_docs,
    description="Use this tool to retrieve relevant documents before answering queries."
)

# Initialize agent
print("initializing agent\n")
agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print("running agent\n")
# Sample query
query = "Tell me the key terms and conditions of the document."
response = agent.run(query)
print(response)