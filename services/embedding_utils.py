import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-ada-002"  # or another available embedding model

def get_embedding(text):
    """
    Call OpenAI to get the embedding for a piece of text.
    Returns a list of floats (the embedding vector).
    """
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    # The returned embedding is in response['data'][0]['embedding']
    embedding_vector = response['data'][0]['embedding']
    return embedding_vector