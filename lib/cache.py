import json
import functools
import os


cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")

os.makedirs(cache_dir, exist_ok=True)


def get_cache_path(name):
    return os.path.join(cache_dir, name)


import json
import os


def buffer(temp_file_name, generator):
    max_retries = 2
    temp_file_path = get_cache_path(temp_file_name)

    for attempt in range(max_retries):
        try:
            if os.path.exists(temp_file_path):
                with open(temp_file_path, "r") as cache_file:
                    for line in cache_file:
                        yield json.loads(line)
                return
            else:
                raise FileNotFoundError(f"Cache file {temp_file_path} not found.")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise e  # If it's the last attempt, re-raise the exception

            print(f"Attempt {attempt + 1} failed: {e}. Regenerating cache...")

            with open(temp_file_path, "w") as cache_file:
                for row in generator:
                    cache_file.write(json.dumps(row) + "\n")


def use_buffer(temp_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            generator = func(*args, **kwargs)
            return buffer(temp_file, generator)

        return wrapper

    return decorator
