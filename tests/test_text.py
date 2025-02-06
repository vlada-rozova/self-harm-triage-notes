from self_harm_triage_notes.text import *
import pandas as pd
from collections import Counter

def test_count_tokens():
    """Test values and type returned by count_tokens()."""
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
    """Execution test for print_token_counts()."""
    x = pd.Series(['I have a tasty apple', 
                   'A red apple is tasty.'])
    print_token_counts(x)

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
            "p{": "p"
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

class TestIsValidToken:
    def test_valid_tokens(self):
        assert is_valid_token("hello") == True
        assert is_valid_token("hello123") == True 
        assert is_valid_token("a") == True
        assert is_valid_token("Test") == True

    def test_invalid_tokens(self):
        assert is_valid_token("123") == False
        assert is_valid_token("") == False
        assert is_valid_token("!@#$") == False
        assert is_valid_token("   ") == False

    def test_edge_cases(self):
        assert is_valid_token("a1") == True
        assert is_valid_token("1a") == True
        assert is_valid_token("a!") == True
        assert is_valid_token("\n") == False
        assert is_valid_token("\t") == False

def test_count_valid_tokens():
    """Test values and type returned by count_valid_tokens()."""
    x = pd.Series(['hello hello world 123', 
                   't.38 120/60'])
    exp_res = Counter({'hello': 2, 'world': 1, 't.38': 1})
    res = count_valid_tokens(x)
    # Check that type is Counter
    assert type(res)==Counter
    # Check that values are correct
    assert res==exp_res


