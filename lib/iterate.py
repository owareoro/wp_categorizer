def iterate(options, eval_func):
    current = {
        key: values[0] for key, values in options.items() if isinstance(values, list)
    }

    while True:
        scores = []
        for key, values in options.items():
            if not isinstance(values, list):
                continue

            if len(values) == 1:
                current[key] = values[0]
                continue

            best_val, best_score = None, float("-inf")
            val_type = type(values[0])

            for val in values:
                current[key] = val
                accuracy = eval_func(current)
                scores.append(accuracy)
                if accuracy >= best_score:
                    best_score = accuracy
                    best_val = val

            if best_val is not None:
                current[key] = best_val
                best_idx = values.index(best_val)

                # Create a new set of values around the best value
                new_values = [best_val]
                if best_idx > 0:
                    new_values.insert(
                        0, val_type((values[best_idx - 1] + best_val) / 2)
                    )
                else:
                    new_values.insert(
                        0, val_type(best_val - (values[best_idx + 1] - best_val))
                    )

                if best_idx < len(values) - 1:
                    new_values.append(val_type((values[best_idx + 1] + best_val) / 2))
                else:
                    new_values.append(
                        val_type(best_val + (best_val - values[best_idx - 1]))
                    )

                # Filter out duplicates if the new range has converged
                new_values = [
                    val
                    for idx, val in enumerate(new_values)
                    if idx == 0 or new_values[idx - 1] != val
                ]

                options[key] = new_values

            print(f"{key} = {best_val} -> {'..'.join(f'{v:.3f}' for v in values)}")
            print(f"Accuracy: {best_score * 100:.2f}%")

        avg_score = sum(scores) / len(scores) * 100 if scores else 0
        min_score = min(scores) * 100 if scores else 0
        print(f"Accuracy: Avg: {avg_score:.2f}% Min: {min_score:.2f}%")
        print(f"Current Best: {current}")
