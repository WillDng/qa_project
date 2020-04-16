import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 


def clean_text_re(text, compiled_re):
    """Returns cleaned text after applying regular expression function"""
    new_text = text
    matches = compiled_re.findall(text)
    for matching_text in matches:
        new_text = new_text.replace(matching_text, '')
    return new_text


class PatternRemover(TransformerMixin, BaseEstimator):
    """Remove matching pattern in text"""

    def __init__(self, compiled_re):
        self.re = compiled_re

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[clean_text_re(text, self.re) 
                  for text in texts],
            columns=[texts.name]
        )


def encoder_re(text, compiled_re):
    """Return 1.0 if match pattern 0.0 otherwise"""
    if compiled_re.findall(text):
        return 1.0
    else:
        return 0.0


class PatternEncoder(TransformerMixin, BaseEstimator):
    """ Encode (binary) a matching pattern in text"""

    def __init__(self, compiled_re):
        self.re = compiled_re

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[encoder_re(text, self.re) 
                  for text in texts],
            columns=[texts.name]
        )


class PatternCounter(TransformerMixin, BaseEstimator):
    """ Encode (binary) a matching pattern in text"""

    def __init__(self, compiled_re):
        self.re = compiled_re

    def count_pattern(self, text):
        """Count pattern in text"""
        return len(self.re.findall(text))

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[self.count_pattern(text) 
                  for text in texts],
            columns=[texts.name]
        )


def correct_mispell(text, reference):
    txt = text.lower().split()
    txt = [(reference[m] if m in reference.keys() else m) for m in text]
    txt = " ".join(txt)
    return txt


class SpellingCorrecter(TransformerMixin, BaseEstimator):
    """Correct spelling following a passed dictionary"""

    def __init__(self, reference_dictionary):
        self.dictionary = reference_dictionary

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[correct_mispell(text, self.dictionary) 
                  for text in texts],
            columns=[texts.name]
        )


class LemmaTfidfVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        lemmatizer = WordNetLemmatizer()
        return lambda doc: [lemmatizer.lemmatize(t) for t in tokenize(doc)]
