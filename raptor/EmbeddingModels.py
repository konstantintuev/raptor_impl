import logging
from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.ERROR)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str):
        pass

    @abstractmethod
    def create_embeddings(self, text: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def create_embedding(self, text):
        return self.model.encode(text)

    def create_embeddings(self, text: List[str]) -> List[List[float]]:
        return self.model.encode(text, convert_to_tensor=False).tolist()
