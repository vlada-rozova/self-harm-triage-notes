from self_harm_triage_notes.text import *
import pandas as pd
from collections import Counter

def test_count_tokens():
    """Test values and type returned by count_tokens"""
    x = pd.Series(['I have a tasty apple', 
                   'A red apple is tasty.'])
    exp_res = Counter({'I': 1, 'have': 1, 'a': 1, 'tasty': 1, 'apple': 2, 
                       'A': 1, 'red': 1, 'is': 1, 'tasty.': 1})
    res = count_tokens(x)
    # Check that type is Counter
    assert type(res)==Counter
    # Check that values are correct
    assert res==exp_res

def test_print_token_counts():
    """Execution test for print_token_counts."""
    x = pd.Series(['I have a tasty apple', 
                   'A red apple is tasty.'])
    print_token_counts(x)

def test_fix_leading_fullstop():
    """Test text is processed correctly by fix_leading_fullstop"""
    text = "i have an apple .the apple is tasty."
    exp_res = "i have an apple . the apple is tasty."
    res = fix_leading_fullstop(text)
    # Check that type is str
    assert type(res)==str
    # Check that text is processed correctly
    assert res==exp_res

