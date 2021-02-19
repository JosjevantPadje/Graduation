import nltk
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import getDatasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

def get_data_target(df, name):
    data = np.array(df['Text'])
    data = (data.reshape(-1, 1)).tolist()
    target = list(df[name])
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=33)
    oversample = RandomOverSampler(sampling_strategy='not majority')
    X_ros, y_ros = oversample.fit_resample(X_train, y_train)
    X_train_flatten = [item for l in X_ros for item in l]
    X_test_flatten = [item for l in X_test for item in l]
    return X_train_flatten, X_test_flatten, y_ros, y_test

def train(X_train, X_test, y_train, y_test, vect, tfidf, clf):
    text_clf = Pipeline([('vect', vect), ('tfidf', tfidf), ('clf', clf)])
    text_clf.fit(X_train, y_train)
    y_score = text_clf.decision_function(X_test)
    print(text_clf.classes_)
    for i, j in enumerate(text_clf.classes_):
        print(j)
        y_score_one = [l[i] for l in y_score]
        print(precision_recall_curve(y_test, y_score_one, pos_label=j))
    #precicion, recall, _ = precision_recall_curve(y_test, y_score, pos_label='koliek')
    #print('precision: ' + str(precicion[0]) + ' at ' + str(recall[0]) + ' recall')
    #return precicion[0]

names_binary_labels = ['hoefbevangen', 'koliek', 'huid', 'luchtweg']
df_labels = pd.read_excel('datasets/labeled.xlsx', encoding='ISO-8859-1', na_filter=False,
                          sheet_name='all_labels_new', index_col=0)

clf = SGDClassifier(random_state=42,loss='modified_huber',alpha= 1e-3,fit_intercept=True,tol=1,shuffle=True,power_t=0.5)
vect = CountVectorizer(ngram_range=(1,1),stop_words=nltk.corpus.stopwords.words('dutch'))
tfidf = TfidfTransformer(use_idf=True,smooth_idf=True,sublinear_tf=True)
disease = 'simpel'
X_train, X_test, y_train, y_test = get_data_target(df_labels, disease)
precision_at_100_recall = train(X_train, X_test, y_train, y_test, vect, tfidf, clf)
#print(precision_at_100_recall)
# d = {}
# for disease in ['hoefbevangen', 'huid', 'luchtweg']:
#     X_train, X_test, y_train, y_test = get_data_target(df_labels, disease)
#     print(disease)
#     d1 = {}
#     d1['start'] = [0]
#     for loss in ['squared_loss', 'hinge', 'squared_hinge', 'modified_huber']:
#         print(loss)
#         for alpha in [1e-1, 1e-2, 1e-3]:
#             for fit_intercept in [True, False]:
#                 for tol in [1e-2, 1e-3, 1e-4]:
#                     for shuffle in [True, False]:
#                         for power_t in [0.5]:
#                             for ngram_range in [(1, 1), (1, 2), (2,2)]:
#                                 for stop_words in [None, nltk.corpus.stopwords.words('dutch')]:
#                                     for use_idf in [False, True]:
#                                         for smooth_idf in [True, False]:
#                                             for sublinear_tf in [True, False]:
#                                                 clf_str = 'SGDClassifier(loss=' + str(loss) + ',alpha=' + str(alpha) + ',fit_intercept=' + str(fit_intercept) + ',tol=' + str(tol) + ',shuffle=' + str(shuffle) + ',power_t=' + str(power_t) + ',random_state=' + str(42) + ')'
#                                                 vect_str = 'SGDClassifier(stop_words=' + str(stop_words) + ',ngram_range=' + str(ngram_range) + ')'
#                                                 tfidf_str = 'TfidfTransformer(use_idf=' + str(use_idf) + ',smooth_idf=' + str(smooth_idf) + ',sublinear_tf=' + str(sublinear_tf) + ')'
#                                                 clf = SGDClassifier(loss=loss, alpha=alpha, fit_intercept=fit_intercept, tol=tol, shuffle=shuffle, power_t=power_t, random_state=42)
#                                                 vect = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
#                                                 tfidf = TfidfTransformer(use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
#                                                 precision_at_100_recall = train(X_train, X_test, y_train, y_test, vect, tfidf, clf)
#                                                 if precision_at_100_recall >= max(d1.values()):
#                                                     print(disease + ': ' + clf_str + vect_str + tfidf_str)
#                                                     print(precision_at_100_recall)
#                                                     d1[clf_str + vect_str + tfidf_str] = precision_at_100_recall
#     d[disease] = d1
#     print(d1)
#
# print(d)



