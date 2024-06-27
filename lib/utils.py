import math
import random


def normalize(word_vec):
    sum_of_squares = sum([freq**2 for freq in word_vec.values()])
    norm = math.sqrt(sum_of_squares)
    for word in word_vec:
        word_vec[word] /= norm


def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[word] * vec2[word] for word in intersection])

    sum1 = sum([vec1[word] ** 2 for word in vec1.keys()])
    sum2 = sum([vec2[word] ** 2 for word in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def get_accuracy(result_set: set, expected_set: set):
    i = len(result_set & expected_set)
    U = len(result_set) + len(expected_set) - i
    return i / U if U > 0 else 0


# Select every n-th element from an iterable
def select(n, iterable):
    try:
        for i in range(random.randint(0, n)):
            next(iterable)
        while True:
            yield next(iterable)
            for i in range(n):
                next(iterable)
    except StopIteration:
        return


def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump
        if y > x and y - x < jump * 0.5 or x > y and x - y < jump * 0.5:
            x = y
