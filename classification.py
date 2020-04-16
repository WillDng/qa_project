import sklearn
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.compose import make_column_transformer

import classification_lib as cl
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
linebreak_re = re.compile(r'\n')
# longueur/verbosité avec nombre de caractères comme proxy
chars_re = re.compile('.')

numbers_re = re.compile(r'\d\.?\d*')
links_re = re.compile(r'www[^\s]*(?=\s)|http[^\s]*(?=\s)')
demonstrations_re = re.compile(r'(?<=\n).*[&\^=\+\_\[\]\{\}\\\|]+.*(?=\n)')

count_encoder_union = make_union(
    cl.PatternCounter(chars_re),
    cl.PatternEncoder(numbers_re),
    cl.PatternEncoder(links_re),
    cl.PatternEncoder(demonstrations_re)
)

full_count_encoder_union = make_union(
    cl.PatternCounter(linebreak_re),
    count_encoder_union
)

question_features = count_encoder_union.transform(X_question)
answer_features = count_encoder_union.transform(X_answer)

cleaner_pipeline = make_pipeline(
    cl.PatternRemover(numbers_re),
    cl.PatternRemover(links_re),
    cl.PatternRemover(demonstrations_re),
    cl.SpellingCorrecter(md.mispell_dict)  
)

ct = make_column_transformer(
    (count_encoder_union, ['question_title']),
    (full_count_encoder_union, ['question_body']),
    (full_count_encoder_union, ['answer']),
    (OneHotEncoder(drop='first'), ['category', 'host'])
)

question_nbchars = X_question.apply(lambda x: len(x))
answer_nbchars = X_answer.apply(lambda x: len(x))
title_nbchars = X_title.apply(lambda x: len(x))

question_numbers = X_question.apply(lambda x: cl.encoder_re(x, numbers_re))   
answer_numbers = X_answer.apply(lambda x: cl.encoder_re(x, numbers_re))
title_numbers = X_title.apply(lambda x: cl.encoder_re(x, numbers_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, numbers_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, numbers_re))



question_links = X_question.apply(lambda x: cl.encoder_re(x, links_re))
answer_links = X_answer.apply(lambda x: cl.encoder_re(x, links_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, links_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, links_re))

question_demonstrations = X_question.apply(lambda x: cl.encoder_re(x, demonstrations_re))
answer_demonstrations = X_answer.apply(lambda x: cl.encoder_re(x, demonstrations_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, demonstrations_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, demonstrations_re))

X_question = X_question.apply(lambda x: md.correct_mispell(x))
X_answer = X_answer.apply(lambda x: md.correct_mispell(x))

# TODO: densité de ponctuation ?
question_mark = re.compile(r'\?')

# OHE avec drop first et au format sparse
ohe = OneHotEncoder(drop='first')
X_category = ohe.fit_transform(X[['category', 'host']])

# TODO: séparer les échantillons d'apprentissage et de test avant 
# les transformations pour éviter trop d'overfitting

stop_words = list(txt.ENGLISH_STOP_WORDS)
for words in swp.stop_words_to_remove:
    stop_words.remove(words)
stop_words += swp.cs_stop_words \
              + swp.generated_during_tokenizing

tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.8,
    ngram_range=(1,2)
)

answer_tfidftransformed = tfidftransformer.fit_transform(X_answer)

# TODO: automatiser la récupératin du seuil ?
n_components = 275
answer_pca = TruncatedSVD(n_components=n_components)
answer_tfidftransformed_acp = answer_pca.fit_transform(answer_tfidftransformed)

# transformation sur le titre et les answers
question_tfidftransformed = tfidftransformer.fit_transform(X_question)

n_components = 251
question_pca = TruncatedSVD(n_components=n_components)
question_tfidftransformed_acp = question_pca.fit_transform(question_tfidftransformed)

title_tfidftransformed = tfidftransformer.fit_transform(X_title)

n_components = 15
title_pca = TruncatedSVD(n_components=n_components)
title_tfidftransformed_acp = title_pca.fit_transform(question_tfidftransformed)

# degré de proximité entre title/answer et question/answer
# TODO : refaire la similarité, les matrices n'ont plus la même taille.
# il faudra probablement récupérer le vocabulaire commun pour fitter et transformer
# title_answer_similarity = cosine_similarity(title_tfidftransformed, 
#                                             answer_tfidftransformed).diagonal()
# title_question_similarity = cosine_similarity(title_tfidftransformed, 
#                                               question_tfidftransformed).diagonal()
# question_answer_similarity = cosine_similarity(question_tfidftransformed, 
#                                                answer_tfidftransformed).diagonal()

X_transformed = pd.concat([
    question_nblines,
    answer_nblines,
    question_nbchars,
    answer_nbchars,
    title_nbchars,
    question_numbers,
    answer_numbers,
    title_numbers,
    question_links,
    answer_links,
    question_demonstrations,
    answer_demonstrations,
    # title_answer_similarity,
    # title_question_similarity,
    # question_answer_similarity,
    title_tfidftransformed_acp,
    question_tfidftransformed_acp, 
    answer_tfidftransformed_acp,
    X_category
])

X_train, X_test, y_train, y_test = train_test_split(X_transformed,
                                                    y_transformed,
                                                    test_size=0.15)

dtc = DecisionTreeClassifier(max_depth=5)
dtc_multi = MultiOutputClassifier(dtc, n_jobs=-1)

dtc_multi.fit(X_train, y_train)
dtc_multi.score(X_test, y_test)

def transform_split_score(X, y):
    if isinstance(X, list):
        X_transformed = sp.hstack(X[text_var].apply(lambda col: vectorizer.fit_transform(col)))
    X_transformed = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed,
                                                        y,
                                                        test_size=0.15)
    dtc_multi.fit(X_train, y_train)
    return dtc_multi.score(X_test, y_test)

# tree interpretation

# feats = {}
# for feature, importance in zip(XXX.columns, dtc.feature_importances_):
#     feats[feature] = importance
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
# importances.sort_values(by='Importance', ascending = False ).head(8)

# affichage de l'arbre de décision

# fig, ax = plt.subplots(figsize=(20, 20))
# sklearn.tree.plot_tree(dtc, ax=ax, filled=True)
