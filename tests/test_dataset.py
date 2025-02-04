import pandas as pd
from self_harm_triage_notes.dataset import *

def test_print_stats():
    """Execution test for print_stats."""
    df = pd.DataFrame({
        'SH': [1,0,0,1],
        'SI': [1,1,1,0],
        'AOD_OD': [0,0,1,0],
    })
    print_stats(df)