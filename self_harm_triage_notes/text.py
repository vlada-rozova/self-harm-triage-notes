from collections import Counter

def count_tokens(x):
    """Count the number of times each unique token occurs in corpus."""
    tokens = []
    x.apply(lambda y: [tokens.append(token) for token in y.split()])
    return Counter(tokens)