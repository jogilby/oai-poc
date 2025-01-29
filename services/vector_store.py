import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim=1536):
        # Initialize an in-memory Faiss index for L2 similarity
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.embeddings = []
        self.text_chunks = []

    def add_embeddings(self, new_embeddings, text_chunks):
        """
        Add new embeddings and their corresponding text chunks to the index.
        """
        if not new_embeddings:
            return
        np_embeddings = np.array(new_embeddings).astype('float32')
        
        # Add to index
        self.index.add(np_embeddings)
        
        # Keep track of text chunks so we can retrieve them later
        self.text_chunks.extend(text_chunks)
        self.embeddings.extend(new_embeddings)
    
    def search(self, query_embedding, k=5):
        """
        Given a query embedding, return the top-k matching text chunks.
        """
        np_query = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(np_query, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Avoid out-of-range index
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], dist))
        return results