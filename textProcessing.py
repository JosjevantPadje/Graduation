from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import nltk
import getDatasets
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline

### print the number of labels in the different lists in a dict ###
def print_label_overview(l, df):
    for i in l:
        print(i, Counter(df[i]))

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
    predicted = text_clf.predict(X_test)
    print('Accuracy: ' + str(np.mean(predicted == y_test)))
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))
    return text_clf

def train_and_label(df, name, parameters_final, df_all):
    X_train, X_test, y_train, y_test = get_data_target(df, name)
    clf = parameters_final[name]['clf']
    vect = parameters_final[name]['vect']
    tfidf = parameters_final[name]['tfidf']
    print(name)
    text_clf = train(X_train, X_test, y_train, y_test, vect, tfidf, clf)
    if name in ['koliek', 'hoefbevangen', 'huid', 'luchtweg']:
        precision_recall(X_test, y_test, text_clf, name)
    else:
        precision_recall_multi(X_test, y_test, text_clf)
    result = text_clf.predict(df_all['Number^Tekst'])
    return result

def precision_recall(X_test, y_test, classifier, name):
    y_score = classifier.decision_function(X_test)#classifier.predict_proba(X_test)
    #y_score_new = [i[1] for i in y_score]
    #print(y_score_new)
    diseases = ['koliek', 'hoefbevangen', 'huid', 'luchtweg']
    if name in diseases:
        precicion, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
        print(name + ': precision: ' + str(precicion[0]) + ' at ' + str(recall[0]) + ' recall')

def precision_recall_multi(X_test, y_test, classifier):
    classes = classifier.classes_
    for i, j in enumerate(classes):
        if j in ['koliek', 'hoefbevangen', 'huid', 'luchtweg']:
            y_test_new = []
            for y in y_test:
                if y == j:
                    y_test_new.append(1)
                else:
                    y_test_new.append(0)
            y_pred = classifier.decision_function(X_test)
            y_pred_new = [l[i] for l in y_pred]
            precicion, recall, _ = precision_recall_curve(y_test_new, y_pred_new, pos_label=1)
            print(j + ': precision: ' + str(precicion[0]) + ' at ' + str(recall[0]) + ' recall')




def main():
    names_all_labels = ['reduced', 'simpel', 'hoefbevangen', 'koliek', 'huid', 'luchtweg']
    names_binary_labels = ['hoefbevangen', 'koliek', 'huid', 'luchtweg']
    df_labels = pd.read_excel('datasets/labeled.xlsx',  na_filter=False,
                                                       sheet_name='all_labels_new', index_col=0) #encoding='ISO-8859-1',
    df_all = getDatasets.get_consults_df()

    parameters_final = {'reduced':{'clf':SGDClassifier(random_state=42,loss='modified_huber',alpha= 1e-3,fit_intercept=True,tol=1,shuffle=False,power_t=0.5),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=nltk.corpus.stopwords.words('dutch')),
                                   'tfidf':TfidfTransformer(use_idf=True,smooth_idf=True,sublinear_tf=True)},
                        'simpel':{ 'clf':SGDClassifier(random_state=42,loss='modified_huber',alpha= 1e-3,fit_intercept=True,tol=1,shuffle=True,power_t=0.5),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=nltk.corpus.stopwords.words('dutch')),
                                   'tfidf':TfidfTransformer(use_idf=True,smooth_idf=True,sublinear_tf=True)},
                        'hoefbevangen':{'clf':SGDClassifier(loss='modified_huber',alpha=0.01,fit_intercept=True,tol=0.0001,shuffle=True,power_t=0.5,random_state=42),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=nltk.corpus.stopwords.words('dutch')),
                                   'tfidf':TfidfTransformer(use_idf=False,smooth_idf=False,sublinear_tf=True)},
                        'koliek': {'clf':SGDClassifier(loss='squared_hinge',alpha=0.001,fit_intercept=False,tol=0.001,shuffle=True,power_t=0.5,random_state=42),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=nltk.corpus.stopwords.words('dutch')),
                                   'tfidf':TfidfTransformer(use_idf=False,smooth_idf=True,sublinear_tf=True)},
                        'huid':{   'clf':SGDClassifier(loss='squared_loss',alpha=0.01,fit_intercept=True,tol=0.0001,shuffle=False,power_t=0.5,random_state=42),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=None),
                                   'tfidf':TfidfTransformer(use_idf=False,smooth_idf=True,sublinear_tf=True)},
                        'luchtweg':{'clf':SGDClassifier(loss='hinge',alpha=0.001,fit_intercept=True,tol=0.01,shuffle=True,power_t=0.5,random_state=42),
                                   'vect':CountVectorizer(ngram_range=(1,1),stop_words=None),
                                   'tfidf':TfidfTransformer(use_idf=False,smooth_idf=False,sublinear_tf=True)}}
    for name in names_all_labels:
        print(name)
        # test_all_parameters(df_labels, name)
        df_all[name] = train_and_label(df_labels, name, parameters_final, df_all)
        if name in ['koliek', 'luchtweg', 'hoefbevangen', 'huid']:
            print(sum(df_all[name]))
        else:
            print(Counter(df_all[name]))
    return df_all

main()