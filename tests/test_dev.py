from self_harm_triage_notes.dev import *
import json
from unittest.mock import mock_open, patch
from collections import Counter

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
