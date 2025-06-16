import pytest
import pandas as pd
from collections import Counter
from unittest.mock import patch, mock_open
from pathlib import Path
from io import StringIO
from self_harm_triage_notes.text_utils import *

def test_is_valid_token():
    """Test basic functionality of is_valid_token()."""
    # Valid tokens
    assert is_valid_token("hello") == True
    assert is_valid_token("hello123") == True 
    assert is_valid_token("a") == True
    assert is_valid_token("Test") == True
    # Invalid tokens
    assert is_valid_token("123") == False
    assert is_valid_token("") == False
    assert is_valid_token("!@#$") == False
    assert is_valid_token("   ") == False
    # Edge cases
    assert is_valid_token("a1") == True
    assert is_valid_token("1a") == True
    assert is_valid_token("a!") == True
    assert is_valid_token("\n") == False
    assert is_valid_token("\t") == False

def test_count_tokens():
    """Test basic functionality of count_tokens()."""
    # Test basic functionality
    test_data = pd.Series([
        'hello world',
        'hello t2dm',
        'hello 37.6'
    ])
    result = count_tokens(test_data)
    expected = Counter({
        'hello': 3,
        'world': 1,
        't2dm': 1,
        '37.6': 1
    })
    assert isinstance(result, Counter)
    assert result == expected

    # Test counting only valid tokens
    result = count_tokens(test_data, valid=True)
    expected = Counter({
        'hello': 3,
        'world': 1,
        't2dm': 1,
    })
    assert isinstance(result, Counter)
    assert result == expected

def test_print_token_counts():
    """Execution test for print_token_counts()."""
    counts = Counter({'I': 1, 'have': 1, 'a': 0, 'tasty': 1, 'apple': 2})
    print_token_counts(counts)

class TestFixLeadingFullstop():
    def test_fix_leading_fullstop_basic(self):
        # Test normal case with leading fullstop
        assert fix_leading_fullstop("word .test") == "word . test"
        # Test multiple leading fullstops
        assert fix_leading_fullstop("word .test .another") == "word . test . another"
        # Test no leading fullstops
        assert fix_leading_fullstop("normal text") == "normal text"

    def test_fix_leading_fullstop_edge_cases(self):
        # Test empty string
        assert fix_leading_fullstop("") == ""
        # Test single fullstop
        assert fix_leading_fullstop(".") == "."
        # Test multiple spaces
        assert fix_leading_fullstop("word   .test") == "word   . test"
        # Test with newlines
        assert fix_leading_fullstop("line\n.test") == "line\n. test"

    def test_fix_leading_fullstop_special(self):
        # Test numbers after fullstop
        assert fix_leading_fullstop("word .123") == "word .123"
        # Test uppercase after fullstop
        assert fix_leading_fullstop("word .TEST") == "word .TEST"
        # Test special characters
        assert fix_leading_fullstop("word .@#$") == "word .@#$"
        # Test multiple fullstops in sequence
        assert fix_leading_fullstop("word ...test") == "word ...test"

class TestPreprocess():
    def test_basic_lowercase(self):
        """Test basic lowercase conversion"""
        assert preprocess("HELLO WORLD") == "hello world"
        assert preprocess("Mixed CASE") == "mixed case"

    def test_special_character_removal(self):
        """Test removal of _x000D_\n"""
        assert preprocess("text_x000D_\nmore text") == "text more text"

    def test_bracket_p_replacement(self):
        """Test replacement of brackets around p"""
        test_cases = {
            "[p": "p",
            "p[": "p",
            "{p": "p",
            "p{": "p",
            "de[pression" : "depression",
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text) == expected

    def test_semicolon_l_replacement(self):
        """Test replacement of semicolon with l"""
        assert preprocess(";l") == "l"
        assert preprocess("l;") == "l"

    def test_backtick_replacement(self):
        """Test replacement of backtick with single quote"""
        assert preprocess("text`s") == "text's"

    def test_direction_expansion(self):
        """Test expansion of l) and r) to left and right"""
        test_cases = {
            "l)": "left",
            "l.": "left .",
            "r)": "right",
            "r.": "right .",
            "word l)": "word left",
            "word r)": "word right"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text) == expected

    def test_symbol_word_replacement(self):
        """Test replacement of symbols with words"""
        test_cases = {
            "time @ place": "time at place",
            "temp ^": "temp elevated",
            "~ 5km": "approximately 5km"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text) == expected

    def test_medical_abbreviations(self):
        """Test medical abbreviation expansions"""
        test_cases = {
            "+ve": "positive",
            "-ve": "negative",
            "co-operative": "cooperative",
            "co operative": "cooperative",
            "r/ship": "relationship",
            "palp'n": "palpitations",
            "med'n": "medication",
            "mov't": "movement"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text) == expected

    def test_medical_terms_normalization(self):
        """Test normalization of medical terms"""
        test_cases = {
            "preg": "pregnant",
            "irreg": "irregular",
            "reg": "regular",
            "rr": "respiratory rate",
            "spo2": "sao2 ",
            "t38": "temperature 38",
            "bp120/80": "blood pressure 120/80",
            "gsc15": "gcs 15"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text).strip() == expected.strip()

    def test_pain_location_normalization(self):
        """Test normalization of pain locations"""
        test_cases = {
            "abdopain": "abdo pain",
            "neckpain": "neck pain",
            "backpain": "back pain",
            "chestpain": "chest pain"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text).strip() == expected.strip()

    def test_punctuation_handling(self):
        """Test handling of various punctuation marks"""
        test_cases = {
            "text....text": "text.text",
            "text///text": "text/text",
            "text---text": "text-text",
            "5-10": "5 - 10",
            "note:text": "note : text",
            "5/min": "5 / min"
        }
        for input_text, expected in test_cases.items():
            assert preprocess(input_text) == expected

    def test_multiple_transformations(self):
        """Test multiple transformations in combination"""
        input_text = "PT @ HOSP w/ BP120/80 & T38.5 +ve for COVID"
        expected = "pt at hosp w/ blood pressure 120/80 & temperature 38.5 positive for covid"
        assert preprocess(input_text) == expected

@pytest.fixture(scope='module')
def nlp():
    return load_nlp_pipeline()

class TestLoadNLPPipeline:    
    def test_basic_pipeline_loading(self, nlp):
        assert nlp is not None
        assert isinstance(nlp, spacy.language.Language)
        
    def test_disabled_components(self, nlp):
        disabled_components = ['tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner']
        for component in disabled_components:
            assert component not in nlp.pipe_names
            
    def test_custom_tokenizer(self, nlp):
        assert nlp.tokenizer.__class__ == spacy.tokenizer.Tokenizer
        # Verify it's using our custom tokenizer
        assert isinstance(nlp.tokenizer, type(combined_rule_tokenizer(nlp)))
        
    def test_removed_rules(self, nlp):
        removed_rules = ['id', 'wed', 'im']
        for rule in removed_rules:
            assert rule not in nlp.tokenizer.rules
            
    def test_tokenization_examples(self, nlp):
        text = "I'm wed to the id concept"
        doc = nlp(text)
        tokens = [token.text for token in doc]
        assert "I" in tokens
        assert "'m" in tokens
        assert "wed" in tokens
        assert "id" in tokens
        
    def test_model_loading_error(self):
        with pytest.raises(OSError):
            # Try loading non-existent model
            spacy.load("non_existent_model")
            
    def test_pipeline_processing(self, nlp):
        text = "Sample medical text"
        doc = nlp(text)
        assert doc.has_annotation("TAG") is False
        assert doc.has_annotation("DEP") is False
        assert doc.has_annotation("ENT_TYPE") is False

def test_doc2str(nlp):
    doc = nlp("I'm a test document")
    assert doc2str(doc) == "i am a test document"

def test_tokenize_step1(nlp):
    x = pd.Series([
                "blood pressure 120/80 temperature 38.5, warm/pink/dry",
                "hr 72 respiratory rate 16",
                "sao2 98; at home w/ pain",
                "patient's gcs 15. w/o visible injuries",
                "patient seed on wed for sob, didn't cooperate"
            ])
    res = tokenize_step1(x)
    # Check that type is str
    assert type(res[0])==str
    # Check that values are correct
    assert res[0]=="blood pressure 120/80 temperature 38.5 , warm/pink/dry"
    assert res[1]=="hr 72 respiratory rate 16"
    assert res[2]=="sao2 98 ; at home w / pain"
    assert res[3]=="patient 's gcs 15 . without visible injuries"
    assert res[4]=="patient seed on wed for sob , do not cooperate"

class TestTokenizeStep2:
    @pytest.fixture(scope='class')
    def sample_vocab(self):
        """Sample vocabulary for testing."""
        return {
            'hello',
            'world',
            'patient',
            'doctor',
            'temp-controlled',  # Example of valid compound token in vocab
            'a/b'              # Another valid compound token
        }

    def test_basic_tokenization(self, sample_vocab):
        """Test basic tokenization with no compound tokens."""
        input_series = pd.Series(['hello world'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'hello world'

    def test_compound_token_split(self, sample_vocab):
        """Test splitting of unknown compound tokens."""
        input_series = pd.Series(['patient-complains'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'patient - complains'

    def test_known_compound_token(self, sample_vocab):
        """Test preservation of known compound tokens."""
        input_series = pd.Series(['temp-controlled room'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'temp-controlled room'

    def test_multiple_compound_tokens(self, sample_vocab):
        """Test handling of multiple compound tokens in one string."""
        input_series = pd.Series(['patient/admits:anxiety'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'patient / admits : anxiety'

    def test_mixed_known_unknown_compounds(self, sample_vocab):
        """Test mixture of known and unknown compound tokens."""
        input_series = pd.Series(['a/b test-case'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'a/b test - case'

    def test_empty_input(self, sample_vocab):
        """Test handling of empty input."""
        input_series = pd.Series([''])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == ''

    def test_no_letters(self, sample_vocab):
        """Test handling of tokens without letters."""
        input_series = pd.Series(['123-456'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == '123-456'

    def test_multiple_rows(self, sample_vocab):
        """Test processing of multiple rows in series."""
        input_series = pd.Series([
            'patient-complains',
            'temp-controlled',
            'normal text'
        ])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'patient - complains'
        assert result.iloc[1] == 'temp-controlled'
        assert result.iloc[2] == 'normal text'

    def test_multiple_separators(self, sample_vocab):
        """Test handling of tokens with multiple separators."""
        input_series = pd.Series(['patient:self/reported-symptoms'])
        result = tokenize_step2(input_series, sample_vocab)
        assert result.iloc[0] == 'patient : self / reported - symptoms'

def test_count_vocab_tokens_in_data():
    """Test values and type returned by count_vocab_tokens_in_data()."""
    x = pd.Series(['hello hello world 123', 
                   't.38 120/80'])
    vocab = {'hello', 'world', 't.38', '120/80'}
    exp_res = Counter({'hello': 2, 'world': 1, 't.38': 1, '120/80': 1})
    res = count_vocab_tokens_in_data(x, vocab)
    # Check that type is Counter
    assert type(res)==Counter
    # Check that values are correct
    assert res==exp_res

class TestCorrectTokens:
    @pytest.fixture(scope="class")
    def _dict(self):
        return {
            'helth': 'health',
            'anxius': 'anxious',
            'suiside': 'suicide',
            'deppresed': 'depressed'
        }

    def test_basic_correction(self, _dict):
        """Test basic spelling correction functionality"""
        text = "patient helth anxius"
        expected = "patient health anxious"
        assert correct_tokens(text, _dict) == expected

    def test_multiple_corrections(self, _dict):
        """Test multiple corrections in one text"""
        text = "helth anxius deppresed"
        expected = "health anxious depressed"
        assert correct_tokens(text, _dict) == expected

    def test_no_corrections_needed(self, _dict):
        """Test text with no misspellings"""
        text = "patient is healthy"
        assert correct_tokens(text, _dict) == text

    def test_empty_text(self, _dict):
        """Test empty string input"""
        assert correct_tokens("", _dict) == ""

    def test_empty_dictionary(self):
        """Test with empty misspelling dictionary"""
        text = "helth anxius"
        assert correct_tokens(text, {}) == text

    def test_mixed_corrections(self, _dict):
        """Test text with both correct and incorrect spellings"""
        text = "patient helth is anxius"
        expected = "patient health is anxious"
        assert correct_tokens(text, _dict) == expected

    def test_multiple_spaces(self, _dict):
        """Test text with multiple spaces between words"""
        text = "helth    anxius"
        expected = "health anxious"
        assert correct_tokens(text, _dict) == expected

def test_select_valid_tokens():
    """Test values and type returned by select_valid_tokens()."""
    text = 'patient presented with t.38 bp 120/60'
    exp_res = 'patient presented with t.38 bp'
    res = select_valid_tokens(text)
    # Check that type is list
    assert type(res)==str
    # Check that values are correct
    assert res==exp_res

def test_load_vocab():
    """Test basic functionality of vocabulary loading with path and filename"""
    # Sample test data
    mock_vocab = ["word1", "word2", "word3"]
    mock_json = json.dumps(mock_vocab)
    
    # Mock file operations
    with patch('builtins.open', mock_open(read_data=mock_json)):
        # Call function with both parameters
        result = load_vocab(Path('test/path'), 'tmp')
        
        # Check results
        assert isinstance(result, frozenset)
        assert len(result) == 3
        assert "word1" in result
        assert "word2" in result
        assert "word3" in result

def test_load_word_list():
    """Test basic functionality of word frequency list loading with path and filename"""
    # Sample test data
    mock_word_list = {'word1': 5, 'word2': 3, 'word3': 2}
    mock_json = json.dumps(mock_word_list)
    
    # Mock file operations
    with patch('builtins.open', mock_open(read_data=mock_json)):
        # Call function with both parameters
        result = load_word_list(Path('test/path'), 'tmp')
        
        # Check results
        assert isinstance(result, Counter)
        assert len(result) == 3
        assert result['word1'] == 5
        assert result['word2'] == 3
        assert result['word3'] == 2
        assert sum(result.values()) == 10


def test_load_misspelled_dict():
    """Test basic functionality of misspelled dict loading with path and filename"""
    # Sample test data
    mock_misspelled_dict = {'word1': 'word4', 'word2': 'word5', 'word3': 'word6'}
    mock_json = json.dumps(mock_misspelled_dict)

    # Mock file operations
    with patch('builtins.open', mock_open(read_data=mock_json)):
        result = load_misspelled_dict(Path('test/path'), 'tmp')
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert result['word1'] == 'word4'
        assert result['word2'] == 'word5'
        assert result['word3'] == 'word6'