import functools
import configparser
from collections import defaultdict
from itertools import chain
import json
import random
from nltk import download
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import math
import pickle

if __name__ == "__main__":
    # Ensure the required NLTK data is downloaded
    download("stopwords")
    download("averaged_perceptron_tagger")
    download("punkt")
    download("wordnet")  # Use nltk downloader to download resource "wordnet"
    download("omw-1.4")  # Use nltk downloader to download resource "omw-1.4"

lemmatizer = WordNetLemmatizer()

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
run_from_database = config.getboolean("Settings", "RunFromDatabase")

# Preprocess text data
stop_words = set(stopwords.words("english"))


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match


# Step 1 - Preprocess Text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    tagged_words = pos_tag(tokens)

    words = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in tagged_words
        if word.isalnum() and word not in stop_words
    ]
    return words


def buffer(temp_file, generator):
    i = 0
    err = None
    while i < 2:
        try:
            with open(temp_file) as fo:
                while True:
                    line = fo.readline()
                    if not line:
                        break
                    yield json.loads(line)
            return
        except Exception as e:
            err = e
            print("Caching results....")
            with open(temp_file, "w") as f:
                for row in generator:
                    f.write(json.dumps(row) + "\n")
            i += 1
    raise err


def use_buffer(temp_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            generator = func(*args, **kwargs)
            return buffer(temp_file, generator)

        return wrapper

    return decorator


@use_buffer(".posts.temp")
def fetch_posts():
    if run_from_database:

        import mysql.connector

        db = None
        try:
            db = mysql.connector.connect(
                host=config.get("Settings", "Host"),
                port=config.get("Settings", "Port"),
                user=config.get("Settings", "User"),
                passwd=config.get("Settings", "Password"),
                database=config.get("Settings", "Database"),
                connect_timeout=60000,
            )
            cursor = db.cursor()
            cursor.execute(
                """
                SELECT ID, post_content, name , taxonomy
                FROM wp_posts 
                JOIN wp_term_relationships ON wp_posts.ID = wp_term_relationships.object_id 
                JOIN wp_term_taxonomy ON wp_term_relationships.term_taxonomy_id = wp_term_taxonomy.term_taxonomy_id
                JOIN wp_terms ON wp_term_taxonomy.term_id = wp_terms.term_id 
                WHERE post_type = 'post' AND (taxonomy = "category" OR taxonomy = "post_tag")
                ORDER BY wp_posts.ID
            """
            )
            last_row = None
            while True:
                print("Reading 200 rows...")
                chunk = cursor.fetchmany(size=200)
                if not chunk:
                    break
                print("Done")
                for row in chunk:
                    cat = "*" + row[2] if row[3] == "category" else row[2]
                    if last_row and row[0] == last_row[0]:
                        last_row[2].append(cat)
                    else:
                        if last_row:
                            yield last_row[1:]
                        last_row = [row[0], row[1], [cat]]
            if last_row:
                yield last_row[1:]
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            if db and db.is_connected():
                cursor.close()
                db.close()
    else:
        placeholder_posts = [
            ("One more freaking post content", ["category2"]),
            ("This is another placeholder post content", ["category1"]),
            ("This is the best post content", ["category3"]),
            ("Final the best post content", ["category3"]),
            ("Some more placeholder post content", ["category1"]),
            ("Another set of freaking post content", ["category2"]),
            ("Some more freaking post content", ["category2"]),
            ("Final placeholder post content", ["category1"]),
            ("This is a freaking post content", ["category2"]),
            ("This is another freaking post content", ["category2"]),
            ("Some more the best post content", ["category3"]),
            ("Yet another freaking post content", ["category2"]),
            ("Yet another placeholder post content", ["category1"]),
            ("This is a placeholder post content", ["category1"]),
            ("Final freaking post content", ["category2"]),
            ("This is another the best post content", ["category3"]),
            ("One more placeholder post content", ["category1"]),
            ("Yet another the best post content", ["category3"]),
            ("Another set of placeholder post content", ["category1"]),
            ("One more the best post content", ["category3"]),
            ("Another set of the best post content", ["category3"]),
        ]
        for post in placeholder_posts:
            yield post


def sigmoid(e):
    return 1 / (1 + math.exp(-e))


def create_defaultdict():
    return defaultdict(float)


def normalize(word_freq):
    sum_of_squares = sum([freq**2 for freq in word_freq.values()])
    norm = math.sqrt(sum_of_squares)
    for word in word_freq:
        word_freq[word] /= norm


def train_model():
    # Term frequency of each word in each category normalized to 1
    category_word_tf = defaultdict(create_defaultdict)
    word_idf = defaultdict(float)
    category_doc_counts = defaultdict(float)
    category_freq = defaultdict(int)
    total_docs = 0

    # Calculate the TF for each word in each category
    for content, categories in fetch_posts():
        total_docs += 1
        words = preprocess_text(content)
        word_freq = defaultdict(float)
        for word in words:
            word_freq[word] += 1
        count = len(words)
        for word, freq in word_freq.items():
            category_doc_counts[word] += 1
            for category in categories:
                category_word_tf[category][word] = freq / count
        for category in categories:
            category_freq[category] += 1

    # Calculate the IDF for each word
    for word in category_doc_counts:
        word_idf[word] = math.log((1 + total_docs) / (1 + category_doc_counts[word]))

    # Calculate the TF-IDF for each word in each category
    for category, word_freq in category_word_tf.items():
        for word, freq in word_freq.items():
            category_word_tf[category][word] *= word_idf[word]
        normalize(word_freq)

    # Normalize the relative popularity of each category
    normalize(category_freq)

    results = (category_word_tf, word_idf, category_freq)
    with open("training_results.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Training completed.")
    return results


def calculate_tfidf(words, word_idf):
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    tfidf_vector = {}
    count = len(words)
    for word, freq in word_freq.items():
        tf = freq / count
        tfidf_vector[word] = tf * word_idf[word]
    normalize(tfidf_vector)
    return tfidf_vector


def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[word] * vec2[word] for word in intersection])

    sum1 = sum([vec1[word] ** 2 for word in vec1.keys()])
    sum2 = sum([vec2[word] ** 2 for word in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


# Categories that start with * always have one selected
# Normally, we could do this in two passes, one for the single categories and another for the tags
def classify_post(
    post, model, cutoff, min_tags=1, max_tags=5, frequency_bias_cutoff=0.15, **_
):
    category_tfidf, word_idf, category_freq = model
    words = preprocess_text(post)
    post_tfidf = calculate_tfidf(words, word_idf)

    cats = []
    tags = []
    for category, category_tfidf in category_tfidf.items():
        similarity = cosine_similarity(post_tfidf, category_tfidf) * max(
            category_freq[category], frequency_bias_cutoff
        )
        if category[0] == "*":
            cats.append((category, similarity))
        else:
            tags.append((category, similarity))

    cats.sort(
        key=lambda e: e[1],
        reverse=True,
    )
    if len(cats) > 1 and (cats[0][0] == "*News" or cats[1][0] == "*News"):
        tags.append(cats[1])
    tags.sort(
        key=lambda e: e[1],
        reverse=True,
    )
    return list(
        chain(
            (cats[0][0],) if cats else tuple(),
            map(
                lambda e: e[0],
                chain(
                    tags[0:min_tags],
                    filter(lambda e: e[1] > cutoff, tags[min_tags:max_tags]),
                ),
            ),
        ),
    )


def get_accuracy(result_set: set, expected_set: set):
    i = len(result_set & expected_set)
    U = len(result_set) + len(expected_set) - i
    return i / U if U > 0 else 0


def load_model():
    try:
        with open("training_results.pickle", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(e)
        return None


def run_classify(content, actual_categories, model, config):
    results = classify_post(content, model, **config)
    return (
        results,
        get_accuracy(
            set(results),
            set(actual_categories),
        ),
        actual_categories,
    )


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


def main():
    # Test the model on the same data
    total_posts = 0
    accuracy = 0
    i = 0
    results_iter = parallel_map(
        run_classify, select(CONFIG["select_size"], fetch_posts()), model, CONFIG
    )
    for results, score, actual_categories in results_iter:
        total_posts += 1
        accuracy += score
        if CONFIG["show_samples"] and (
            i < 10 or i < 100 and i % 10 == 0 or i < 1000 and i % 100 == 0
        ):
            print(f"{i}\tActual Categories:\t{actual_categories}")
            print(f" \tPredicted Categories:\t{results}, Score: {score}")
            print("")
        i += 1

    return (accuracy / total_posts) if total_posts > 0 else 0


def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump
        if y > x and y - x < jump * 0.5 or x > y and x - y < jump * 0.5:
            x = y


CONFIG = {
    "cutoff": 0.06,
    "min_tags": 0,
    "max_tags": 5,
    "select_size": 0,
    "frequency_bias_cutoff": 0.25,
    "show_samples": True,
}  # Gotten from training

shared_args = {}


def init(task_id, func, args):
    if task_id in shared_args:
        raise Exception("Already initialized")
    shared_args[task_id] = (func, args)


def star_run(e):
    func, args = shared_args[e[0]]
    return func(*e[1], *args)


def parallel_map(func, generator, *args):
    import random
    from multiprocessing import Pool

    pickle.dumps(func)
    pickle.dumps(args)

    task_id = random.randint(0, 1000000)
    with Pool(
        initializer=init,
        initargs=(
            task_id,
            func,
            args,
        ),
    ) as p:
        yield from p.imap(
            star_run,
            map(
                lambda e: (task_id, (e if hasattr(e, "__iter__") else (e,))),
                generator,
            ),
            chunksize=20,
        )


def iterate(config, eval_func):
    for i in config:
        if type(config[i]) == list:
            CONFIG[i] = config[i][0]
    while True:
        scores = []
        for i in config:
            if type(config[i]) != list:
                continue
            tries = config[i]
            if len(tries) == 1:
                config[i] = CONFIG[i] = tries[0]
                continue
            val_type = type(tries[0])
            best_val = None
            best = 0
            for j in tries:
                CONFIG[i] = j
                accuracy = eval_func()
                scores.append(accuracy)
                if accuracy >= best:
                    best = accuracy
                    best_val = j
            if best_val is not None:
                CONFIG[i] = best_val
                x = tries.index(best_val)
                m = [best_val]
                if x > 0:
                    m.insert(0, val_type((tries[x - 1] + best_val) / 2))
                else:
                    m.insert(0, val_type(tries[x] - (tries[x + 1] - best_val)))
                if x < len(tries) - 1:
                    m.append(val_type((tries[x + 1] + best_val) / 2))
                else:
                    m.append(val_type(tries[x] + (best_val - tries[x - 1])))
                if m[0] == m[1]:
                    m = m[1:]
                if m[-2] == m[-1]:
                    m = m[:-1]
                config[i] = m
            print(f"{i} = {best_val} -> {'..'.join(map(lambda e: '%.3f' % e, tries))}")
            print(f"Accuracy: {best * 100:.2f}%")
        print(
            f"Accuracy: Avg: {sum(scores)/len(scores) * 100:.2f} Min: {min(scores) * 100:.2f}"
        )
        print(CONFIG)


if __name__ == "__main__":
    print("Starting training...")
    CONFIG["select_size"] = 1
    model = load_model() or train_model()
    # CONFIG["show_samples"] = False
    # CONFIG["select_size"] = 5
    # iterate(
    #     {
    #         "cutoff": [0, 0.1],
    #         "min_tags": [1, 2],
    #         "max_tags": [3, 8],
    #         "frequency_bias_cutoff": [0.1, 0.3],
    #     },
    #     main,
    # )
    main()
