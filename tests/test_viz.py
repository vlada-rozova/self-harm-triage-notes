from self_harm_triage_notes.viz import *

def test_format_quarter():
    """Test basic quarter to month conversion"""
    assert format_quarter('2020Q1') == 'Jan 2020'
    assert format_quarter('2021Q2') == 'Apr 2021'
    assert format_quarter('2022Q3') == 'Jul 2022'
    assert format_quarter('2024Q4') == 'Oct 2024'