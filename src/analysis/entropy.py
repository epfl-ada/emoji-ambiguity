from scipy.stats import entropy


def get_emoji_entropy(x):
    total = sum(x.values(), 0.0)
    for key in x:
        x[key] /= total
    return entropy(list(x.values()))
