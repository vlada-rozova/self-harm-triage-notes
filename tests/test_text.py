from self_harm_triage_notes.text import *
import pandas as pd
from collections import Counter

def test_count_tokens():
    """Test values and type returned by count_tokens"""
    x = pd.Series(['I have a tasty apple', 
                   'A red apple is tasty.'])
    res = count_tokens(x)
    exp_res = Counter({'I': 1, 'have': 1, 'a': 1, 'tasty': 1, 'apple': 2, 
                       'A': 1, 'red': 1, 'is': 1, 'tasty.': 1})
    # Check that type is Counter
    assert type(res)==Counter
    # Check that values are correct
    assert res==exp_res

def test_print_token_counts():
    """Execution test for print_token_counts."""
    x = pd.Series(['I have a tasty apple', 
                   'A red apple is tasty.'])
    print_token_counts(x)
