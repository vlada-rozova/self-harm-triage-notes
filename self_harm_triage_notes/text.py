from collections import Counter
import re

def count_tokens(x):
    """Count the number of times each unique token occurs in corpus."""
    tokens = []
    x.apply(lambda y: [tokens.append(token) for token in y.split()])
    return Counter(tokens)

def print_token_counts(x):
    """Print stats for token counts."""
    counts = count_tokens(x)
    print("The corpus contains %d unique tokens (%d tokens in total)." % (len(counts), sum(counts.values())))

def fix_leading_fullstop(text):
    """
    Separate a leading full stop with a whitespace.
    """
    pattern = re.compile("(?<=\s)\.(?=[a-z])")
    text = pattern.sub(r". ", text)
    return text

def preprocess(text):
    """
    Apply custom pre-processing to ED triage notes. 
    """
    # Convert to lower case
    text = text.lower()
    
    # Remove _x000D_\n
    text = text.replace("_x000d_\n", r" ")
    
    # Remove "[" or "{" where meant p
    pattern = re.compile("\[p|p\[|{p|p{")
    text = pattern.sub(r"p", text)
    
    # Remove ";" where meant l
    pattern = re.compile(";l|l;")
    text = pattern.sub(r"l", text)
    
    # Replace "`" with "'"
    text = text.replace("`", r"'")
    
    # "l)" to "left"
    pattern = re.compile("\sl(?=[\)\.])")
    text = pattern.sub(r" left ", text)
    pattern = re.compile("^l(?=[\)\.])")
    text = pattern.sub(r" left ", text)
    
    # "r)" to "right"
    pattern = re.compile("\sr(?=[\)\.])")
    text = pattern.sub(r" right ", text)
    pattern = re.compile("^r(?=[\)\.])")
    text = pattern.sub(r" right ", text)
    
    # "@" to "at"
    text = text.replace("@", r" at ")
    
    # "^" to "elevated"
    text = text.replace("^", r" elevated ")
    
    # "~" to "approximately"
    text = text.replace("~", r" approximately ")

    # "#" to "fractured" if not followed by number
    pattern = re.compile("#(?!\d)")
    text = pattern.sub(r" fracture ", text)
    
    # "+ve" to "positive"
    pattern = re.compile("\+ve(?![a-z])")
    text = pattern.sub(r" positive ", text)
    
    # "-ve" to "positive"
    pattern = re.compile("\-ve(?![a-z])")
    text = pattern.sub(r" negative ", text)
    
    # "co operative" and "co-operative" to "cooperative"
    pattern = re.compile("co[\s-]operative")
    text = pattern.sub(r"cooperative", text)
    
    # "r/ship" to relationship
    pattern = re.compile("r/ships?")
    text = pattern.sub(r" relationship ", text)
    
    # "palp'n" to "palpitation"
    pattern = re.compile("palp\'ns?")
    text = pattern.sub(r" palpitations ", text)
    
    # "med'n" to "medication"
    pattern = re.compile("med\'ns?")
    text = pattern.sub(r" medication ", text)
    
    # "mov't" to "movement
    pattern = re.compile("mov\'ts?")
    text = pattern.sub(r" movement ", text)
    
    # 1. Replace "preg" by "pregnant"
    pattern = re.compile("preg$|preg\.?(\W)")
    text = pattern.sub(r" pregnant \1", text)
    
    # 2. Replace "reg" by "regular"
    pattern = re.compile("irreg$|irreg\.?(\W)")
    text = pattern.sub(r" irregular \1", text)
    pattern = re.compile("reg$|reg\.?(\W)")
    text = pattern.sub(r" regular \1", text)
    
    # 3. Normalise respiratory rate
    pattern = re.compile("([^a-z]|^)rr(?![a-z])|resp\srate|resp\W?(?=\d)")
    text = pattern.sub(r"\1 respiratory rate ", text)
    
    # 4. Normalise oxygen saturation
    pattern = re.compile("sp\s?[o0]2|sp2|spo02|sa\s?[o0]2|sats?\W{0,3}(?=\d)")
    text = pattern.sub(r" sao2 ", text) 
    pattern = re.compile("([^a-z])sp\W{0,3}(?=[19])")
    text = pattern.sub(r"\1 oxygen saturation ", text)
    
    # 5. Normilise temperature
    pattern = re.compile("([^a-z]|^)t(emp)?\W{0,3}(?=[34]\d)")
    text = pattern.sub(r"\1 temperature ", text)

    # 6. Normalise hours
    pattern = re.compile("([^a-z])hrs|([^a-z])hours")
    text = pattern.sub(r"\1 hours ", text)
     
    # 7. Normalise heart rate
    pattern = re.compile("([^a-z])hr(?![a-z])")
    text = pattern.sub(r"\1 heart rate ", text)
    
    # 8. Normalise GCS
    text = text.replace("gsc", r"gcs")
    
    # 9. Normalise on arrival
#     pattern = re.compile("o/a|on arrival|on assessment")
#     text = pattern.sub(r" on arrival ", text)
    
    # 10. Normalise abdo/neck/back/chest
    pattern = re.compile("(abdo|neck|back|chest)pain")
    text = pattern.sub(r" \1 pain ", text)
    
    # 11. Normalise section 351
    pattern = re.compile("(section|sect|s)\s?351")
    text = pattern.sub(r" section351 ", text)

    # Add spaces around "bp"
    pattern = re.compile("([^a-z]|^)bp(?![a-z])")
    text = pattern.sub(r"\1 blood pressure ", text)
    
    # Add spaces around "bmp", "bsl", "gcs"
    pattern = re.compile("(bpm|bsl|gcs)")
    text = pattern.sub(r" \1 ", text)
    
    # Remove some punctuation marks
    pattern = re.compile("[!#$%()*,<=>?@[\\]^_`{|}~]")
    text = pattern.sub(r" ", text)
    
    # Remove duplicated punctuation marks "&'+-./:;
    pattern = re.compile('\"{2,}')
    text = pattern.sub(r'"', text)
    
    pattern = re.compile("&{2,}")
    text = pattern.sub(r"&", text)
    
    pattern = re.compile("\'{2,}")
    text = pattern.sub(r"'", text)
    
    pattern = re.compile("\+{2,}")
    text = pattern.sub(r"+", text)
    
    pattern = re.compile("-{2,}")
    text = pattern.sub(r"-", text)
    
    pattern = re.compile("\.{2,}")
    text = pattern.sub(r".", text)
    
    pattern = re.compile("/{2,}")
    text = pattern.sub(r"/", text)
    
    pattern = re.compile(":{2,}")
    text = pattern.sub(r":", text)
    
    pattern = re.compile(";{2,}")
    text = pattern.sub(r";", text)  
    
    # Remove "." and "'" where meant "/"
    pattern = re.compile("\./|/\.|\'/|/\'")
    text = pattern.sub(r"/", text)
    
    # Add spaces around - when digits on one or both sides
    pattern = re.compile("(?<=\d)-|-(?=\d)")
    text = pattern.sub(r" - ", text)

    # Add spaces around : when letters on one or both sides
    pattern = re.compile("(?<=[a-z]):|:(?=[a-z])")
    text = pattern.sub(r" : ", text)
    
    # Add spaces around "/" when digit on one side and alpha on the other
    pattern = re.compile("(?<=\d)/(?=[a-z])|(?<=[a-z])/(?=\d)")
    text = pattern.sub(r" / ", text)
    
    # Separate two punctuation marks with a space
    pattern = re.compile("([\"&'+-./:;])(?=[\"&'+-./:;])")
    text = pattern.sub(r"\1 ", text)
    
    # Add a space after a leading "."
    text = fix_leading_fullstop(text)
    
    # Remove duplicated spaces
    pattern = re.compile("\s{2,}")
    text = pattern.sub(r" ", text)

    # Strip
    text = text.strip()
    
    return text

def is_valid_token(token):
    """
    Check if a token contains any letters. 
    """
    for ch in token:
        if ch.isalpha():
            return True
    return False