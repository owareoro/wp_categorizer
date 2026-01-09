import numpy as np
import random
from collections import defaultdict


def normalize(word_vec):
    """
    Normalize a word vector using the Euclidean norm (L2 norm).
    """
    norm = np.linalg.norm(word_vec)
    if norm == 0:
        return word_vec  # Avoid division by zero
    return word_vec / norm


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two word vectors.
    """
    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0
    return float(numerator) / denominator


def get_accuracy(result_vec, expected_vec):
    """
    Calculate the Jaccard similarity between the result and expected vectors.
    """
    intersection = np.logical_and(result_vec, expected_vec).sum()
    union = np.logical_or(result_vec, expected_vec).sum()
    return intersection / union if union > 0 else 0


def select(n, iterable):
    """
    Select every n-th element from an iterable, starting from a random offset.
    """
    try:
        iterator = iter(iterable)
        for _ in range(random.randint(0, n)):
            next(iterator)
        while True:
            yield next(iterator)
            for _ in range(n):
                next(iterator)
    except StopIteration:
        return


def frange(x, y, jump):
    """
    Generate a range of floating-point numbers from x to y with a specified step (jump).
    """
    return np.arange(x, y + jump, jump)


def create_defaultdict():
    return defaultdict(float)


def get_default_dict():
    return defaultdict(create_defaultdict)
