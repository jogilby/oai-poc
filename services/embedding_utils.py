import openai
import asyncio
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-ada-002"

async def async_get_embedding(text):
    """
    Async function to get the embedding using OpenAI's async client.
    """
    response = await openai.Embedding.acreate(
        model=EMBEDDING_MODEL,
        input=text
    )
    # The embedding is in response["data"][0]["embedding"]
    return response["data"][0]["embedding"]

def get_embedding(text):
    """
    Synchronous wrapper that blocks on the async call.
    """
    return asyncio.run(async_get_embedding(text))