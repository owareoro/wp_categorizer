import configparser
from collections import defaultdict
from itertools import chain
from nltk import download
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import math
from lib.loader import load_model, save_model
from lib.parallel import parallel_map
from lib.cache import use_buffer
from lib.utils import cosine_similarity, get_accuracy, select, normalize
from lib.iterate import iterate
from lib.stats import RunningStats

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


def get_synonyms(word):
    synsets = wn.synsets(word)
    if not synsets:
        return set()
    res = set()
    for synset in synsets:
        res.add(synset.lemmas()[0].name())
    return res


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


running_stats = RunningStats()


def add_synonyms(word_freq):
    for word in word_freq:
        synonyms = get_synonyms(word)
        if len(synonyms) == 0:
            return
        word_freq[word] *= 0.5
        each = word_freq[word] / len(synonyms)
        for synonym in synonyms:
            word_freq[synonym] += each


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
        if total_docs % 50 == 0:
            print(f"Processed {total_docs} documents.")

    # Calculate the IDF for each word
    for word in category_doc_counts:
        word_idf[word] = math.log((1 + total_docs) / (1 + category_doc_counts[word]))

    # Calculate the TF-IDF for each word in each category
    for category, word_freq in category_word_tf.items():
        for word, freq in word_freq.items():
            word_freq[word] *= word_idf[word]
        running_stats.add(word_idf[word])
        add_synonyms(word_freq)
        normalize(word_freq)

    # Normalize the relative popularity of each category
    normalize(category_freq)

    results = (category_word_tf, word_idf, category_freq)
    save_model(results)
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


def main(conf={}):
    CONFIG.update(conf)
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


CONFIG = {
    "cutoff": 0.06,
    "min_tags": 0,
    "max_tags": 5,
    "select_size": 0,
    "frequency_bias_cutoff": 0.25,
    "show_samples": True,
}  # Gotten from training


if __name__ == "__main__":
    print("Starting training...")
    CONFIG["select_size"] = 1
    model = load_model() or train_model()
    print(
        f"Stats: mean: {running_stats._mean} variance: {running_stats._variance}",
    )
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
