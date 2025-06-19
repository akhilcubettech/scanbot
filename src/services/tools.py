from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma

from src.services.chroma import client
from src.services.embedding import embedding_fn

load_dotenv()

chroma = Chroma(
    client=client,
    collection_name="scandlearn",
    embedding_function=embedding_fn,
)

retriever = chroma.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "score_threshold": 0.7, "fetch_k": 30}
)

retriever_tool = create_retriever_tool(retriever=retriever,
                                       name="search",
                                       description="search and retrieve documents based on query")

