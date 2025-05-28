import os

from google.genai import Client
from google.oauth2 import service_account
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field


class GenAIEmbeddings(Embeddings, BaseModel):
    """Embeddings implementation using `google-genai`."""

    model_name: str = Field(default="text-multilingual-embedding-002")
    client: Client = Field(
        default_factory=lambda: Client(
            credentials=service_account.Credentials.from_service_account_file(
                filename=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        ),
        exclude=True,
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings using the Google GenAI API.

        Args:
            texts (list[str]): The text strings to embed.

        Returns:
            list[list[float]]: The embedding vectors.
        """
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
            )
            assert (
                response.embeddings is not None
            ), "No embeddings found in the response."
            for embedding in response.embeddings:
                assert (
                    embedding.values is not None
                ), "No embedding values found in the response."
                embeddings.append(embedding.values)
        assert len(embeddings) == len(
            texts
        ), "The number of embeddings does not match the number of texts."
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a text string using the Google GenAI API.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings using the Google GenAI API asynchronously.

        Args:
            texts (list[str]): The text strings to embed.

        Returns:
            list[list[float]]: The embedding vectors.
        """
        embeddings = []
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        assert response.embeddings is not None, "No embeddings found in the response."
        for embedding in response.embeddings:
            assert (
                embedding.values is not None
            ), "No embedding values found in the response."
            embeddings.append(embedding.values)
        assert len(embeddings) == len(
            texts
        ), "The number of embeddings does not match the number of texts."
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a text string using the Google GenAI API asynchronously.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        return (await self.aembed_documents([text]))[0]
