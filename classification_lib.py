import re

def clean_text_re(text, compiled_re):
    """Returns cleaned text after applying regular expression function"""
    new_text = text
    matches = compiled_re.findall(text)
    for matching_text in matches:
        new_text = text.replace(matching_text, '')
    return new_text


def encoder_re(text, compiled_re):
    """Return 1.0 if match pattern 0.0 otherwise"""
    if compiled_re.findall(text):
        return 1.0
    else:
        return 0.0

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]