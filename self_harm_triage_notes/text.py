from collections import Counter

def count_tokens(x):
    """Count the number of times each unique token occurs in corpus."""
    tokens = []
    x.apply(lambda y: [tokens.append(token) for token in y.split()])
    return Counter(tokens)

def print_token_counts(x):
    """Print stats for token counts."""
    counts = count_tokens(x)
    print("The corpus contains %d unique tokens (%d tokens in total)." % (len(counts), sum(counts.values())))