import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import re
import joblib
import sklearn.feature_extraction.text as txt

from paths import joblib_dir
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.compose import make_column_transformer

import classification_lib as cl
import cosine_normalisation_pipeline as cnp
import stop_words_perso as swp 
import mispell_dict as md

train = pd.read_csv('data/train.csv')

y = train.iloc[:, 11:]

# transformation des targets variables catégorielles
y_transformed = y.apply(lambda x: pd.cut(x,
                        [-0.1, .25, .5, .75, 1.1],
                        labels=['low', 'medium-', 'medium+', 'high']))

# séparation en cas d'études séparés sur les questions ou answers
y_question = y_transformed.loc[:, y_transformed.columns.str.startswith('question')]
y_answer = y_transformed.loc[:, y_transformed.columns.str.startswith('answer')]

to_delete_var = ['qa_id', 'url',
                 'question_user_name', 'question_user_page',
                 'answer_user_name', 'answer_user_page']

X = train.iloc[:, :11].drop(to_delete_var, 1)
X_title = train.question_title
X_question = train.question_body
X_answer = train.answer

# nombre de lignes avec passage à la ligne comme proxy
linebreak_re = r'\n'
# longueur/verbosité avec nombre de caractères comme proxy
chars_re = r'.'

numbers_re = r'\d\.?\d*'
links_re = r'www[^\s]*(?=\s)|http[^\s]*(?=\s)'
demonstrations_re = r'(?<=\n).*[&\^=\+\_\[\]\{\}\|]+.*(?=\n)'
belonging_re = r'\'s'
# TODO: densité de ponctuation ?
# question_mark = r'\?'


count_encoder_union = make_union(
    cl.PatternCounter(chars_re),
    cl.PatternEncoder(numbers_re),
    cl.PatternEncoder(links_re),
    cl.PatternEncoder(demonstrations_re),
    verbose=True
)

full_count_encoder_union = make_union(
    cl.PatternCounter(linebreak_re),
    count_encoder_union,
    verbose=True
)

cleaner_pipeline = make_pipeline(
    cl.PatternRemover(numbers_re),
    cl.PatternRemover(links_re),
    cl.PatternRemover(demonstrations_re),
    cl.PatternRemover(belonging_re),
    cl.SpellingCorrecter(),
    verbose=True
)

cleaner_count_encoder_ct = make_column_transformer(
    ('passthrough', ['question_title']),
    (cleaner_pipeline, ['question_body']),
    (cleaner_pipeline, ['answer']),
    ('passthrough', ['category', 'host']),
    (count_encoder_union, ['question_title']),
    (full_count_encoder_union, ['question_body']),
    (full_count_encoder_union, ['answer']),
    remainder='drop',
    verbose=True
)

X_transformed = pd.DataFrame(
    data=cleaner_count_encoder_ct.fit_transform(train),
    columns=[
        'question_title', 'question_body', 'answer',
        'category', 'host',
        'title_chars',  'title_num', 'title_links', 'title_demo',
        'question_linebreak', 'question_chars', 'question_num', 
        'question_links', 'question_demo',
        'answer_linebreak', 'answer_chars', 'answer_num', 
        'answer_links', 'answer_demo'
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y_transformed,
    test_size=0.15
)

stop_words = list(txt.ENGLISH_STOP_WORDS)
for words in swp.stop_words_to_remove:
    stop_words.remove(words)
stop_words += swp.cs_stop_words \
              + swp.generated_during_tokenizing

title_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.85,
    ngram_range=(1,2)
)

question_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.85,
    ngram_range=(1,2)
)

answer_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.85,
    ngram_range=(1,2)
)

title_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    title_tfidftransformer,
    TruncatedSVD(n_components=15),
    verbose=True
)

question_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    question_tfidftransformer,
    TruncatedSVD(n_components=220),
    verbose=True
)

answer_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    answer_tfidftransformer,
    TruncatedSVD(n_components=250),
    verbose=True
)

cat_host_ohe = OneHotEncoder(drop='first', sparse=False)

tfidf_ohe_ct = make_column_transformer(
    (title_tfidf_acp_pipe, 0),
    (question_tfidf_acp_pipe, 1),
    (answer_tfidf_acp_pipe, 2),
    (cat_host_ohe, [3,4]),
    verbose=True,
    remainder='passthrough'
)

X_train_transformed = tfidf_ohe_ct.fit_transform(X_train).astype(float)

cosine_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    ngram_range=(1,2)
)

cosine_tfidftransformer.fit(
    X_train.question_title
    + ' ' + X_train.question_body
    + ' ' + X_train.answer
)

X_train_transformed = cnp.do_and_stack_cosine(
    cosine_tfidftransformer,
    X_train_transformed,
    X_train
)
