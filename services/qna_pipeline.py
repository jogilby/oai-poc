import openai
import asyncio
from .embedding_utils import get_embedding
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
CHAT_MODEL = "gpt-3.5-turbo"  # or "gpt-4"

async def async_generate_answer(query, retrieved_chunks):
    """
    Async function to call OpenAI ChatCompletion.
    """
    # Combine top retrieved chunks to form context
    context_text = "\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are a helpful assistant for question answering based on context. "
        "Use only the context below to answer. If you do not find the answer in "
        "the context, say you are unsure.\n\n"
        f"Context:\n{context_text}\n\n"
    )
    user_prompt = f"Question: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = await openai.ChatCompletion.acreate(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer

def generate_answer(query, retrieved_chunks):
    """
    Synchronous wrapper around async_generate_answer.
    """
    return asyncio.run(async_generate_answer(query, retrieved_chunks))


def answer_query(vector_store, query, top_k=3):
    """
    Synchronous function that:
      1) Embeds the query
      2) Retrieves top-k chunks
      3) Generates final answer using the LLM
    """
    # Step 1: Get query embedding (internally calls async embedding function)
    query_embedding = get_embedding(query)

    # Step 2: Retrieve top-k chunks
    results = vector_store.search(query_embedding, k=top_k)
    top_chunks = [r[0] for r in results]

    # Step 3: Generate final answer (internally calls async chat function)
    answer = generate_answer(query, top_chunks)
    return answer