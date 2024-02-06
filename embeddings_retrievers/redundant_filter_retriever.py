from typing import Optional, List, Dict, Any

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # Calc embeddings for the query string
        emb = self.embeddings.embed_query(query)
        # Take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )

    async def aget_relevant_documents(self):
        return []
