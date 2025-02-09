from self_harm_triage_notes.config import spell_corr_dir
import json
from collections import Counter


def load_vocab(filename):
    """
    Load vocabulary.
    """
    with open (spell_corr_dir / (filename + "_vocab.json"), 'rb') as f:
        vocab = json.load(f)
        
    print("Domain-specific vocabulary contains %d words." % len(vocab))
    
    return frozenset(vocab)

def load_word_list(filename):
    """
    Load word frequency list.
    """
    with open (spell_corr_dir / (filename + "_word_freq_list.json"), 'rb') as f:
        word_list = json.load(f)
        
    print("Word frequency list contains %d unique words (%d in total)." % 
          (len(word_list), sum(word_list.values())))
    
    return Counter(word_list)

def load_misspelled_dict(filename):
    """
    Load dictionary of misspellings.
    """
    with open (spell_corr_dir / (filename + "_misspelled_dict.json"), 'rb') as f:
        misspelled_dict = json.load(f)
    
    print("Spelling correction available for %d words." % len(misspelled_dict))
        
    return misspelled_dict