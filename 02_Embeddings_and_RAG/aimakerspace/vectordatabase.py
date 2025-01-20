import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

# Implementing euclidean distance, manhattan distance, chebyshev distance, and jaccard similarity
# For my assignment, I will be using jaccard similarity
'''
The "better" metric depends entirely on the context of your problem, the type of data you are working with, and the specific requirements of your analysis or algorithm. Here's a comparison of the four methods:

---

### 1. **Euclidean Distance**
- **Formula**: \( \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \)
- **Properties**:
  - Measures straight-line ("as-the-crow-flies") distance.
  - Works well for continuous, numeric data.
- **Strengths**:
  - Intuitive and widely used in geometric and clustering tasks (e.g., k-means).
  - Sensitive to the magnitude of differences.
- **Limitations**:
  - Sensitive to scale—features with larger ranges dominate the distance.
  - Assumes all dimensions are equally important.
- **Best for**:
  - Continuous numeric data where the magnitude of differences matters.

---

### 2. **Manhattan Distance (City Block Distance)**
- **Formula**: \( \sum_{i=1}^n |x_i - y_i| \)
- **Properties**:
  - Measures distance along axes at right angles (like navigating a grid).
  - More robust to outliers than Euclidean distance.
- **Strengths**:
  - Works well for data with high-dimensional or sparse features.
  - Less sensitive to large outliers compared to Euclidean.
- **Limitations**:
  - May not align well with natural distances in some geometries.
- **Best for**:
  - Data with high dimensionality or a grid-like structure (e.g., urban planning, taxi routes).

---

### 3. **Chebyshev Distance**
- **Formula**: \( \max(|x_i - y_i|) \)
- **Properties**:
  - Measures the greatest single-axis difference between two points.
  - Useful in environments where movement is restricted to one axis at a time.
- **Strengths**:
  - Captures the maximum variation in any one dimension.
  - Intuitive for some grid-based or discrete data scenarios.
- **Limitations**:
  - Ignores variations in other dimensions beyond the maximum.
- **Best for**:
  - Scenarios with uniform costs across axes or where maximum deviations matter (e.g., chessboard distances).

---

### 4. **Jaccard Similarity**
- **Formula**: \( \frac{|A \cap B|}{|A \cup B|} \)
- **Properties**:
  - Measures the similarity between two sets.
  - Specifically designed for binary or categorical data.
- **Strengths**:
  - Works well with binary or set data.
  - Focuses on the proportion of shared elements over total elements.
- **Limitations**:
  - Cannot measure differences in magnitude.
  - Sensitive to the sparsity of data.
- **Best for**:
  - Comparing sets, binary vectors, or categorical data (e.g., text/document similarity).

---

### Key Considerations
- **Data Type**:
  - Continuous: Euclidean or Manhattan.
  - Binary/Categorical: Jaccard.
  - Discrete/Grid: Chebyshev.
- **Scale Sensitivity**:
  - Normalize data if using Euclidean or Manhattan to avoid bias.
- **Dimensionality**:
  - High-dimensional data often benefits from Manhattan or Jaccard due to reduced sensitivity to the curse of dimensionality.

---

In summary:
- Use **Euclidean Distance** for continuous numeric data and clustering if magnitude matters.
- Use **Manhattan Distance** for sparse or high-dimensional data.
- Use **Chebyshev Distance** for maximum deviation-based scenarios (e.g., grids).
- Use **Jaccard Similarity** for comparing sets, binary features, or categorical data.
'''


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))

def chebyshev_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Chebyshev distance between two vectors."""
    return np.max(np.abs(vector_a - vector_b))

def jaccard_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Jaccard similarity between two vectors."""
    intersection = np.sum(np.logical_and(vector_a, vector_b))
    union = np.sum(np.logical_or(vector_a, vector_b))
    return intersection / union

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = jaccard_similarity,
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = jaccard_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
