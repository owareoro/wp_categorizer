import random
from multiprocessing import Pool, get_context

shared_args = {}


def init(task_id, func, args):
    if task_id in shared_args:
        raise ValueError(f"Task ID {task_id} is already initialized.")
    shared_args[task_id] = (func, args)


def star_run(e):
    func, args = shared_args[e[0]]
    return func(*e[1], *args)


def parallel_map(func, generator, *args, chunksize=20):
    task_id = random.randint(0, 1000000)

    with get_context("spawn").Pool(
        initializer=init, initargs=(task_id, func, args)
    ) as pool:
        task_generator = (
            (task_id, e if hasattr(e, "__iter__") else (e,)) for e in generator
        )
        yield from pool.imap(star_run, task_generator, chunksize=chunksize)
