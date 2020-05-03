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
from sklearn.preprocessing import StandardScaler
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

# nombre de lignes avec passage à la ligne comme proxy
linebreak_re = r'\n'
# longueur/verbosité avec nombre de caractères comme proxy
chars_re = r'.'

numbers_re = r'\d\.?\d*'
links_re = r'www[^\s]*(?=\s)|http[^\s]*(?=\s)'
demonstrations_re = r'(?<=\n).*[&\^=\+\_\[\]\{\}\|]+.*(?=\n)'
belonging_re = r'\'s'

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

count_encoder_ct_headers = [
        'title_chars',  'title_num', 'title_links', 'title_demo',
        'question_linebreak', 'question_chars', 'question_num', 
        'question_links', 'question_demo',
        'answer_linebreak', 'answer_chars', 'answer_num', 
        'answer_links', 'answer_demo'
]

X_transformed = pd.DataFrame(
    data=cleaner_count_encoder_ct.fit_transform(train),
    columns=[
        'question_title', 'question_body', 'answer',
        'category', 'host',
    ] + count_encoder_ct_headers
)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    train.iloc[:, 11:],
    test_size=0.15
)

y_train_transformed = cl.discretize_targets(y_train)
y_test_transformed = cl.discretize_targets(y_test)

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

title_acp = TruncatedSVD(n_components=15)

title_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    title_tfidftransformer,
    title_acp,
    verbose=True
)

question_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.85,
    ngram_range=(1,2)
)

question_acp = TruncatedSVD(n_components=220)

question_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    question_tfidftransformer,
    question_acp,
    verbose=True
)

answer_tfidftransformer = cl.LemmaTfidfVectorizer(
    sublinear_tf=True,
    stop_words=stop_words,
    min_df=0.015,
    max_df=0.85,
    ngram_range=(1,2)
)

answer_acp = TruncatedSVD(n_components=250)

answer_tfidf_acp_pipe = make_pipeline(
    cl.Squeezer(),
    answer_tfidftransformer,
    answer_acp,
    verbose=True
)

cat_host_ohe = OneHotEncoder(
    sparse=False,
    handle_unknown='ignore')

tfidf_ohe_ct = make_column_transformer(
    (title_tfidf_acp_pipe, 0),
    (question_tfidf_acp_pipe, 1),
    (answer_tfidf_acp_pipe, 2),
    (cat_host_ohe, [3,4]),
    verbose=True,
    remainder='passthrough'
)

X_train_transformed = tfidf_ohe_ct.fit_transform(X_train).astype(float)

pipelines_headers = cl.get_pipelines_feature_names(
    tfidf_ohe_ct,
    [
        ('pipeline-1', 'title'),
        ('pipeline-2', 'question'),
        ('pipeline-3', 'answer')
    ]
)

ohe_headers = cl.get_ohe_headers(tfidf_ohe_ct)

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

cosine_headers = [
    'cos_title_question',
    'cos_title_answer',
    'cos_question_answer'
]

columns_headers = pipelines_headers\
                  + ohe_headers\
                  + count_encoder_ct_headers\
                  + cosine_headers

X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=columns_headers
)

X_test_transformed = cnp.do_and_stack_cosine(
    cosine_tfidftransformer,
    tfidf_ohe_ct.transform(X_test),
    X_test
)

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=columns_headers
)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import RegressorChain
from xgboost import XGBClassifier

clf_dtc = MultiOutputClassifier(
    DecisionTreeClassifier(
        class_weight='balanced'
    ),
    n_jobs=-1
)
clf_rfc = MultiOutputClassifier(
    RandomForestClassifier(
        class_weight='balanced_subsample',
        n_jobs=-1,
        oob_score=True,
        max_depth=5,
        max_features='sqrt'
    ),
    n_jobs=-1
)

clf_xgb = MultiOutputClassifier(
    XGBClassifier(
        objective='multi:softmax',
        n_jobs=-1,
        verbosity=1,
        num_class=10,
        booster='gbtree',
        importance_type='gain',
        scale_pos_weight=0.8,
        max_depth=5,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
    n_jobs=-1
)

param_grid_dtc  = [
    {
        'estimator__min_samples_leaf': [1, 3, 5],
        'estimator__max_features': ['sqrt', 'log2']
    }
]

param_grid_rfc  = [
    {
        'estimator__n_estimators': [10, 150, 300, 500],
        'estimator__min_samples_leaf': [1, 5, 10],
    }
]

param_grid_xgb  = [
    {
        'estimator__learning_rate': [0.1, 0.6],
        'estimator__n_estimators': [10, 300, 500]
    }
]

gridcvs={}

for pgrid, clf, name in zip(
    (param_grid_dtc, param_grid_rfc, param_grid_xgb),
    (clf_dtc, clf_rfc, clf_xgb),
    ('multi_dtc', 'multi_rfc', 'multi_xgb')
):
    gcv = GridSearchCV(clf,
                       pgrid,
                       cv=3,
                       refit=True,
                       scoring=cl.custom_accu_score,
                       n_jobs=-1,
                       verbose=True)
    gridcvs[name] = gcv

outer_cv = KFold(n_splits=3, shuffle=True)
outer_scores = {}

for name, gs in gridcvs.items():

    nested_score = cross_val_score(
        gs, 
        X_train_transformed, 
        y_train_transformed, 
        cv=outer_cv,
        scoring=cl.custom_accu_score,
        n_jobs=-1,
        verbose=1
    )
    outer_scores[name] = nested_score
    print(f'{name}: outer accuracy {100*nested_score.mean():.2f} +/- {100*nested_score.std():.2f}')

selected_model = gridcvs['multi_rfc']
selected_model.fit(
    X_train_transformed,
    y_train_transformed
)

print(selected_model.best_params_)

####

clf_rfc = MultiOutputClassifier(
    RandomForestClassifier(
        class_weight='balanced_subsample',
        n_jobs=-1,
        n_estimators=300,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=5,
    ),
    n_jobs=-1
)

clf_rfc.fit(
    X_train_transformed,
    y_train_transformed
)


#####
import shap
import joblib

shap_output = dict()

for index, feat in enumerate(y_train.columns):
    explainer = shap.TreeExplainer(clf_rfc.estimators_[index])
    shap_values = explainer.shap_values(
        X_test_transformed,
        check_additivity=False        
    )
    joblib.dump(shap_values, 'joblib/'+'_'.join(['shap', feat]))
    print('Done', index, feat)
    shap_output[feat] = shap_values

# pour les analyses globales
shap.summary_plot(
    shap_output['answer_type_instructions'],
    X_test_transformed,
    class_names=clf_rfc.classes_[-4]
)

# pour les analyses de classe
shap.summary_plot(
    shap_output['answer_type_instructions'][-1],
    X_test_transformed
)