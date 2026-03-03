import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_query_chunk_similarity(query_embedding, chunk_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), chunk_embeddings)
    return similarities.flatten()


def compute_chunk_redundancy(chunk_embeddings):
    matrix = cosine_similarity(chunk_embeddings)
    redundancy_pairs = []

    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0.9:
                redundancy_pairs.append((i, j, matrix[i][j]))

    return redundancy_pairs