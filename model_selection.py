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

import classification_lib as cl
import stop_words_perso as swp 
import mispell_dict as md

train = pd.read_csv('train.csv')

y = train.iloc[:, 11:]

# transformation des targets variables catégorielles
#y_transformed = y.apply(lambda x: pd.cut(x,
#                        [-0.1, .25, .5, .75, 1.1],
#                        labels=['low', 'medium-', 'medium+', 'high']))
y_transformed=y

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
title_numbers = X_title.apply(lambda x: cl.encoder_re(x, numbers_re))

X_question = X_question.apply(lambda x: cl.clean_text_re(x, numbers_re))
X_answer = X_answer.apply(lambda x: cl.clean_text_re(x, numbers_re))

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

X_transformed=pd.DataFrame()
for col, label in [(question_nblines, 'question_nblines'),
                    (answer_nblines, 'answer_nblines'),
                    (question_nbchars, 'question_nbchars'),
                    (answer_nbchars, 'answer_nbchars'),
                    (title_nbchars, 'title_nbchars'),
                    (question_numbers, 'question_numbers'),
                    (answer_numbers, 'answer_numbers'),
                    (title_numbers, 'title_numbers'),
                    (question_links, 'question_links'),
                    (answer_links, 'answer_links'),
                    (question_demonstrations, 'question_demonstrations'),
                    (answer_demonstrations, 'answer_demonstrations')]:
    X_transformed[label]=col
    
X_transformed = X_transformed.merge(pd.DataFrame(title_tfidftransformed_acp), 
                                    on=X_transformed.index).drop('key_0', axis=1)
X_transformed = X_transformed.merge(pd.DataFrame(question_tfidftransformed_acp), 
                                    on=X_transformed.index).drop('key_0', axis=1)
X_transformed = X_transformed.merge(pd.DataFrame(answer_tfidftransformed_acp), 
                                    on=X_transformed.index).drop('key_0', axis=1)
X_transformed = X_transformed.merge(pd.DataFrame(X_category.toarray()), 
                                    on=X_transformed.index).drop('key_0', axis=1)

l=[]

for col in list(y_transformed.columns):
    for item in y_transformed[col].unique():
        l.append(item)
set(l)

# test Multiple models

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=.2, random_state=111)

clf_rfr = RandomForestRegressor(random_state=0)

param_grid_rfr = [{'n_estimators': [10, 50, 100],
                   'min_samples_leaf': [1, 3, 5],
                   'max_features': ['sqrt', 'log2']}]


clf_chain = RegressorChain(RandomForestRegressor(random_state=0), order=None, cv=None, random_state=0)

param_grid_chain = [{'base_estimator__n_estimators': [10, 50, 100],
                   'base_estimator__min_samples_leaf': [1, 3, 5],
                   'base_estimator__max_features': ['sqrt', 'log2']}]

gridcvs={}

for pgrid, clf, name in zip((param_grid_rfr,
                             param_grid_chain),
                            (clf_rfr, 
                             clf_chain),
                            ('RFR', 'chained_RFR')):
    gcv = GridSearchCV(clf,
                       pgrid,
                       cv=3,
                       refit=True)
    gridcvs[name] = gcv


outer_cv = KFold(n_splits=3, shuffle=True)
outer_scores = {}

for name, gs in gridcvs.items():
    nested_score = cross_val_score(gs, 
                                   X_train, 
                                   y_train, cv=outer_cv)
    outer_scores[name] = nested_score
    
outer_scores

chain = gridcvs['chained_RFR']
chain.fit(X_train, y_train)

chain.best_params_

rfr = gridcvs['RFR']
rfr.fit(X_train, y_train)

import numpy as np 
from scipy import stats

y_pred = rfr.predict(X_test)
corrs=[]
for col in range(len(y_test.columns)):
    corr = stats.spearmanr(pd.DataFrame(y_pred).iloc[:,col], y_test.iloc[:,col])
    corrs.append(corr.correlation)

mean_spearman = np.mean(corrs)

mean_spearman

'''

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import RegressorChain


X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=.2, random_state=111)


clf_rfr = RandomForestRegressor(random_state=0)

param_grid_rfr = [{'n_estimators': [10, 50, 100],
                   'min_samples_leaf': [1, 3, 5],
                   'max_features': ['sqrt', 'log2']}]


clf_chain = RegressorChain(RandomForestRegressor(random_state=0), order=None, cv=None, random_state=0)

param_grid_chain = [{'base_estimator__n_estimators': [10, 50, 100],
                   'base_estimator__min_samples_leaf': [1, 3, 5],
                   'base_estimator__max_features': ['sqrt', 'log2']}]


gcv = GridSearchCV(clf_chain,param_grid_chain,cv=3, refit=True)

y_train = y_train.iloc[:, 0:3]
y_test = y_test.iloc[:,0:3]

gcv.fit(X_train, y_train)
y_pred = gcv.predict(X_train)
'''