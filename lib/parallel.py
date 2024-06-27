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
