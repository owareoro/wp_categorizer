import pickle
from lib.cache import get_cache_path


def load_model():
    try:
        with open(get_cache_path("training_results.pickle"), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(e)
        return None


def save_model(model):
    with open(get_cache_path("training_results.pickle"), "wb") as f:
        pickle.dump(model, f)
