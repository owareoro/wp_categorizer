def iterate(options, eval_func):
    current = {}
    for i in options:
        if type(options[i]) == list:
            current[i] = options[i][0]
    while True:
        scores = []
        for i in options:
            if type(options[i]) != list:
                continue
            tries = options[i]
            if len(tries) == 1:
                options[i] = current[i] = tries[0]
                continue
            val_type = type(tries[0])
            best_val = None
            best = 0
            for j in tries:
                current[i] = j
                accuracy = eval_func(current)
                scores.append(accuracy)
                if accuracy >= best:
                    best = accuracy
                    best_val = j
            if best_val is not None:
                current[i] = best_val
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
                options[i] = m
            print(f"{i} = {best_val} -> {'..'.join(map(lambda e: '%.3f' % e, tries))}")
            print(f"Accuracy: {best * 100:.2f}%")
        print(
            f"Accuracy: Avg: {sum(scores)/len(scores) * 100:.2f} Min: {min(scores) * 100:.2f}"
        )
        print(current)
