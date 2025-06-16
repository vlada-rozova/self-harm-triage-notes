import pytest
import spacy
from self_harm_triage_notes.custom_tokenizer import *

@pytest.fixture
def nlp():
    nlp = spacy.load("en_core_sci_sm", 
                     disable=['tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner'])
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    return nlp

class TestRemoveNewLines:
    @pytest.mark.parametrize("input_text,expected", [
        ("word-\n\nbreak", "wordbreak"),
        ("word- \n\nbreak", "wordbreak"),
        ("first-\n\nbreak second-\nbreak", "firstbreak secondbreak"),
        ("normal text", "normal text"),
        ("", ""),
        ("-\n\n", ""),
    ])
    def test_newline_removal(self, input_text, expected):
        assert remove_new_lines(input_text) == expected

class TestCombinedRulePrefixes:
    def test_prefix_patterns(self):
        patterns = combined_rule_prefixes()
        expected_basics = ["§", "%", "=", r"\+"]
        for pattern in expected_basics:
            assert pattern in patterns
    
    def test_bracket_patterns(self):
        patterns = combined_rule_prefixes()
        assert any(r"\((?![^\(\s]+\)\S+)" in p for p in patterns)
        assert any(r"\[(?![^\[\s]+\]\S+)" in p for p in patterns)
        assert any(r"\{(?![^\{\s]+\}\S+)" in p for p in patterns)

class TestCombinedRuleTokenizer:
    def test_basic_tokenization(self, nlp):
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        doc = nlp("Hello world!")
        assert len(doc) == 3
        assert [t.text for t in doc] == ["Hello", "world", "!"]
    
    @pytest.mark.parametrize("text,expected_tokens", [
        ("x123", ["x", "123"]),
        ("test-case", ["test-case"]),
        ("$100", ["$", "100"]),
        ("John's", ["John", "'s"]),
    ])
    def test_special_cases(self, nlp, text, expected_tokens):
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        doc = nlp(text)
        assert [t.text for t in doc] == expected_tokens
    
    def test_number_handling(self, nlp):
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        doc = nlp("123x456 45.67 1D.")
        assert [t.text for t in doc] == ["123", "x", "456", "45.67", "1D", "."]
    
    def test_bracket_handling(self, nlp):
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        doc = nlp("(test) [example] {text}")
        assert [t.text for t in doc] == ["(", "test", ")", "[", "example", "]", "{", "text", "}"]