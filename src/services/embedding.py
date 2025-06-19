import os

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()


class CustomOpenAIEmbeddings(OpenAIEmbeddings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _embed_documents(self, texts):
        return super().embed_documents(texts)

    def __call__(self, query):
        return self._embed_documents(query)


embedding_fn = CustomOpenAIEmbeddings(model=str(os.getenv("EMBED_MODEL")), api_key=str(os.getenv("OPENAI_API_KEY")))
