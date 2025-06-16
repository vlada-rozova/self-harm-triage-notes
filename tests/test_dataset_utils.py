import pandas as pd
from self_harm_triage_notes.dataset_utils import *

def test_get_mapping():
    """Execution test for get_mapping."""
    mapping = get_mapping()
    assert isinstance(mapping, dict)
    assert mapping[0] == 'Negative'
    assert mapping[1] == 'Positive'

def test_print_stats():
    """Execution test for print_stats."""
    df = pd.DataFrame({
        'SH': [1,0,0,1],
        'SI': [1,1,1,0],
        'AOD_OD': [0,0,1,0],
    })
    print_stats(df)