import numpy as np
from scipy import stats
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import make_scorer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import mispell_dict as md 

def clean_text_re(text, pattern):
    """Returns cleaned text after applying regular expression function"""
    new_text = text
    matches = re.compile(pattern).findall(text)
    if not matches:
        return new_text
    for matching_text in matches:
        new_text = new_text.replace(matching_text, ' ')
    return new_text


class PatternRemover(TransformerMixin, BaseEstimator):
    """Remove matching pattern in text"""

    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[clean_text_re(text, self.pattern) 
                  for text in texts.squeeze()],
            columns=texts.columns
        )


def encoder_re(text, pattern):
    """Return 1.0 if match pattern 0.0 otherwise"""
    if re.compile(pattern).findall(text):
        return 1.0
    else:
        return 0.0


class PatternEncoder(TransformerMixin, BaseEstimator):
    """ Encode (binary) a matching pattern in text"""

    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[encoder_re(text, self.pattern) 
                  for text in texts.squeeze()],
            columns=[texts.columns]
        )


def count_pattern(text, pattern):
    """Count pattern in text"""
    return len(re.compile(pattern).findall(text))


class PatternCounter(TransformerMixin, BaseEstimator):
    """ Encode (binary) a matching pattern in text"""

    # def __init__(self, pattern):
        # self.pattern = pattern
    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[count_pattern(text, self.pattern) 
                  for text in texts.squeeze()],
            columns=texts.columns
        )

reference = md.mispell_dict

def correct_mispell(text):
    text = text.lower().split()
    text = [reference[m] if m in reference.keys() else m for m in text]
    text = " ".join(text)
    return text


class SpellingCorrecter(TransformerMixin, BaseEstimator):
    """Correct spelling following a passed dictionary"""

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        return pd.DataFrame(
            data=[correct_mispell(text) for text in texts.squeeze()],
            columns=[texts.columns]
        )


class Squeezer(TransformerMixin, BaseEstimator):
    """Transform a panda dataframe to series"""

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        return df.squeeze()


class LemmaTfidfVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        lemmatizer = WordNetLemmatizer()
        return lambda doc: [lemmatizer.lemmatize(t) for t in tokenize(doc)]


def format_cosine(array):
    """Format cosine result in order to hstack"""
    return np.expand_dims(array, axis=1).astype(float)


def spearman_score(
    y_true, 
    y_pred, 
    **kwargs
):
    spearman_corr = stats.spearmanr(
        y_true,
        y_pred,
        axis=1
    ).correlation
    return np.mean(spearman_corr)

custom_spearman_score = make_scorer(
    spearman_score,
    greater_is_better=True
)

def accu_score(
    y_true, 
    y_pred, 
    **kwargs
):
    corresp = y_pred == y_true
    return corresp.sum().sum() / (y_true.shape[0] * y_true.shape[1])

custom_accu_score = make_scorer(
    accu_score,
    greater_is_better=True
)
