import json
import functools
import os


cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")

os.makedirs(cache_dir, exist_ok=True)


def get_cache_path(name):
    return os.path.join(cache_dir, name)


def buffer(_temp_file, generator):
    temp_file = get_cache_path(_temp_file)
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
