from self_harm_triage_notes.config import N_SPLITS
from self_harm_triage_notes.dev import *
import json
from unittest.mock import mock_open, patch
from collections import Counter
from sklearn.model_selection._split import StratifiedKFold

def test_load_vocab():
    mock_vocab_data = ['word1', 'word2', 'word3']
    mock_json = json.dumps(mock_vocab_data)
    with patch('builtins.open', mock_open(read_data=mock_json)):
        vocab = load_vocab('test')
        assert isinstance(vocab, frozenset)
        assert len(vocab) == 3
        assert 'word1' in vocab
        assert 'word2' in vocab
        assert 'word3' in vocab

def test_load_word_list():
    mock_word_list_data = {'word1': 5, 'word2': 3, 'word3': 2}
    mock_json = json.dumps(mock_word_list_data)
    with patch('builtins.open', mock_open(read_data=mock_json)):
        word_list = load_word_list('test')
        assert isinstance(word_list, Counter)
        assert len(word_list) == 3
        assert word_list['word1'] == 5
        assert word_list['word2'] == 3
        assert word_list['word3'] == 2
        assert sum(word_list.values()) == 10

def test_load_misspelled_dict():
    mock_misspelled_dict_data = {'word1': 'word4', 'word2': 'word5', 'word3': 'word6'}
    mock_json = json.dumps(mock_misspelled_dict_data)
    with patch('builtins.open', mock_open(read_data=mock_json)):
        misspelled_dict = load_misspelled_dict('test')
        assert isinstance(misspelled_dict, dict)
        assert len(misspelled_dict) == 3
        assert misspelled_dict['word1'] == 'word4'
        assert misspelled_dict['word2'] == 'word5'
        assert misspelled_dict['word3'] == 'word6'

def test_get_stopwords():
    stopwords = get_stopwords()
    assert isinstance(stopwords, frozenset)
    assert len(stopwords) > 0
    assert 'the' in stopwords
    assert 'and' in stopwords
    assert 'or' in stopwords

def test_get_cv_strategy():
    cv = get_cv_strategy()
    # Check that type is StratifiedKFold
    assert type(cv)==StratifiedKFold
    # Check the number of splits, random state, and shuffle
    assert cv.n_splits==N_SPLITS
    assert cv.random_state==3
    assert cv.shuffle==True