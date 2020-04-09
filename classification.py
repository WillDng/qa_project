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

import classification_lib as cl
import stop_words_perso as swp 

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

# extraction du nombre de lignes avec passage à la ligne comme proxy
linebreak_re = re.compile(r'\\n')
question_nblines = X_question.apply(lambda x: len(linebreak_re.findall(x)))
answer_nblines = X_question.apply(lambda x: len(linebreak_re.findall(x)))

# extraction du nombre de la longueur/verbosité avec nombre de caractères comme proxy
question_nbchars = X_question.apply(lambda x: len(x))
answer_nbchars = X_answer.apply(lambda x: len(x))
title_nbchars = X_title.apply(lambda x: len(x))

numbers_re = re.compile(r'\d\.*\d*') 

question_numbers = X_question.apply(lambda x: cl.encoder_re(x, numbers_re))   
answer_numbers = X_answer.apply(lambda x: cl.encoder_re(x, numbers_re))

link_re = re.compile(r'www[^\s]*(?=\s)|http[^\s]*(?=\s)')

question_links = X_question.apply(lambda x: cl.encoder_re(x, link_re))
answer_links = X_answer.apply(lambda x: cl.encoder_re(x, link_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, link_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, link_re))

demonstration_re = re.compile(r'(?<=\n).*[&\^=\+\_\[\]\{\}\\\|]+.*(?=\n)')

question_demonstrations = X_question.apply(lambda x: cl.encoder_re(x, demonstration_re))
answer_demonstrations = X_answer.apply(lambda x: cl.encoder_re(x, demonstration_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, demonstration_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, demonstration_re))


question_mark = re.compile(r'\?')

stop_words = list(txt.ENGLISH_STOP_WORDS)
for words in swp.stop_words_to_remove:
    stop_words.remove(words)

# OHE avec drop first et au format sparse
ohe = OneHotEncoder(drop='first', sparse=True)
X_category = ohe.fit_transform(X[['category', 'host']])

# TF-IDF transformer sans stop-words pour l'instant
answer_tfidftransformer = TfidfVectorizer(sublinear_tf=True,
                                          tokenizer=cl.LemmaTokenizer(),
                                          stop_words=stop_words)

# fit sur les answers (possède plus de vocabulaire)
answer_tfidftransformed = answer_tfidftransformer.fit_transform(X_answer)

# transformation sur le titre et les answers
question_tfidftransformed = answer_tfidftransformer.transform(X_question)
title_tfidftransformed = answer_tfidftransformer.transform(X_title)

# degré de proximité entre title/answer et question/answer
title_answer_similarity = cosine_similarity(title_tfidftransformed, answer_tfidftransformed).diagonal()
title_question_similarity = cosine_similarity(title_tfidftransformed, question_tfidftransformed).diagonal()
question_answer_similarity = cosine_similarity(question_tfidftransformed, answer_tfidftransformed).diagonal()

try:
    question_acp = joblib.load(joblib_dir+'question_features_pca_3000.joblib')
    answer_acp = joblib.load(joblib_dir+'answer_features_pca_4000.joblib')
except FileNotFoundError:
    raise('jblib file not found, run script first')

question_tfidftransformed_acp = question_acp.transform(question_tfidftransformed)
answer_tfidftransformed_acp = answer_acp.transform(answer_tfidftransformed)

X_transformed = sp.hstack([question_tfidftransformed_acp, 
                           answer_tfidftransformed_acp,
                           X_category])

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
