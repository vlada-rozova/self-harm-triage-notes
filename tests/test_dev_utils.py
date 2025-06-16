from self_harm_triage_notes.dev_utils import *

def test_get_stopwords():
    stopwords = get_stopwords()
    assert isinstance(stopwords, list)
    assert len(stopwords) > 0
    assert 'the' in stopwords
    assert 'and' in stopwords
    assert 'or' in stopwords
