from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import getDatasets
import numpy as np

def get_data_target(df):
    array = df.values
    i = len(df.columns)-1
    data = array[:, 0:i]
    target = array[:, i]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=33)
    return X_train, X_test, y_train, y_test

def bagging(X_train, y_train, cart, X_test, y_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10)
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(X_train, y_train)
    results = model.score(X_test, y_test)
    y_df = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    precicions, recall, t = precision_recall_curve(y_test, y_df, pos_label=1)
    print(precicions[:10], recall[:10], t[:10])
    precision = precicions[0]
    confmat = confusion_matrix(y_test, y_pred)
    return results, precision, confmat

def boosting(X_train, y_train, X_test, y_test):
    seed = 7
    num_trees = 100
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed).fit(X_train, y_train) # of model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model.score(X_test, y_test)
    y_df = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    precicions, recall, t = precision_recall_curve(y_test, y_df, pos_label=1)
    print(precicions[:10], recall[:10], t[:10])
    precision = precicions[0]
    confmat = confusion_matrix(y_test, y_pred)
    return results, precision, confmat

def voting(X_train, y_train, estimators, X_test, y_test): ## trains with all models !!
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    ensemble = VotingClassifier(estimators).fit(X_train, y_train)
    results = ensemble.score(X_test, y_test)
    y_df = ensemble.transform(X_test)
    y_pred = ensemble.predict(X_test)
    precicions, recall, t = precision_recall_curve(y_test, y_df, pos_label=1)
    print(precicions[:10], recall[:10], t[:10])
    precision = precicions[0]
    confmat = confusion_matrix(y_test, y_pred)
    return results, precision, confmat

def default(X_train, X_test, y_train, y_test, model):
    fitted = model.fit(X_train, y_train)
    results = fitted.score(X_test, y_test)
    y_df = fitted.decision_function(X_test)
    y_pred = fitted.predict(X_test)
    precicions, recall, t = precision_recall_curve(y_test, y_df, pos_label=1)
    print(precicions[:10], recall[:10], t[:10])
    precision = precicions[0]
    confmat = confusion_matrix(y_test, y_pred)
    return results, precision, confmat

def main():
    colic_df = getDatasets.get_labeled_weather_colic_df()
    laminitis_df = getDatasets.get_labeled_weather_laminitis_df()
    respiratory_df = getDatasets.get_labeled_weather_respiratory_df()
    skin_df = getDatasets.get_labeled_weather_skin_df()
    data_target = [('colic', colic_df), ("laminitis", laminitis_df), ("respiratory", respiratory_df), ("skin", skin_df)]

    model1 = LogisticRegression(max_iter=500)
    model2 = SVC()
    model3 = DecisionTreeClassifier()
    model4 = MLPClassifier()
    estimators = [("LR", model1), ("SVM", model2), ("DT", model3), ("NN", model4)]
    res = []
    for i, d in data_target:
        X_train, X_test, y_train, y_test = get_data_target(d)
        #results = voting(X_train, y_train, estimators, X_test, y_test)
        #print(i + ' voting')
        #print(results)
        #res.append(('voting', i, results))
        results = boosting(X_train, y_train,  X_test, y_test)
        print(i + ' boosting')
        print(results)
        res.append(('boosting', i, results))
        for j, m in estimators:
            results = default(X_train, X_test, y_train, y_test, m)
            res.append(('single' + j, i, results))
            print(i + ' single ' + j)
            print(results)
            results = bagging(X_train, y_train, m, X_test, y_test)
            res.append(('bagging ' + j, i, results))
            print(i + ' bagging ' + j)
            print(results)
    print(res)


main()


l = [('voting', 'colic', (0.6833759590792838, 0.3209718670076726, np.array([[2627,   28],
       [1210,   45]]))), ('boosting', 'colic', (0.6920716112531969, 0.3209718670076726, np.array([[2536,  119],
       [1085,  170]]))), ('singleLR', 'colic', (0.6774936061381074, 0.3209718670076726, np.array([[2569,   86],
       [1175,   80]]))), ('bagging LR', 'colic', (0.6795396419437341, 0.3209718670076726, np.array([[2572,   83],
       [1170,   85]]))), ('singleSVM', 'colic', (0.6790281329923273, 0.3209718670076726, np.array([[2655,    0],
       [1255,    0]]))), ('bagging SVM', 'colic', (0.6790281329923273, 0.3209718670076726, np.array([[2655,    0],
       [1255,    0]]))), ('singleDT', 'colic', (0.6368286445012787, 0.3209718670076726, np.array([[1932,  723],
       [ 697,  558]]))), ('bagging DT', 'colic', (0.7010230179028133, 0.3209718670076726, np.array([[2385,  270],
       [ 899,  356]]))), ('singleNN', 'colic', (0.46342710997442454, 0.3209718670076726, np.array([[ 966, 1689],
       [ 409,  846]]))), ('bagging NN', 'colic', (0.6782608695652174, 0.3209718670076726, np.array([[2651,    4],
       [1254,    1]]))),('voting', 'laminitis', (0.6496973032471106, 0.3483764446890479, np.array([[2341,   27],
       [1246,   20]]))), ('boosting', 'laminitis', (0.6463951568519538, 0.3483764446890479, np.array([[2202,  166],
       [1119,  147]]))), ('singleLR', 'laminitis', (0.6408915795266924, 0.3483764446890479, np.array([[2174,  194],
       [1111,  155]]))), ('bagging LR', 'laminitis', (0.6450192625206385, 0.3483764446890479, np.array([[2197,  171],
       [1119,  147]]))), ('singleSVM', 'laminitis', (0.6516235553109522, 0.3483764446890479, np.array([[2368,    0],
       [1266,    0]]))), ('bagging SVM', 'laminitis', (0.6516235553109522, 0.3483764446890479, np.array([[2368,    0],
       [1266,    0]]))), ('singleDT', 'laminitis', (0.596587782058338, 0.3483764446890479, np.array([[1657,  711],
       [ 755,  511]]))), ('bagging DT', 'laminitis', (0.6496973032471106, 0.3483764446890479, np.array([[2119,  249],
       [1024,  242]]))), ('singleNN', 'laminitis', (0.6516235553109522, 0.3483764446890479, np.array([[2367,    1],
       [1265,    1]]))), ('bagging NN', 'laminitis', (0.6444689047881123, 0.3483764446890479, np.array([[2245,  123],
       [1169,   97]]))), ('voting', 'respiratory', (0.7984654731457801, 0.20153452685421994, np.array([[3122,    0],
       [ 788,    0]]))), ('boosting', 'respiratory', (0.7946291560102302, 0.20153452685421994, np.array([[3092,   30],
       [ 773,   15]]))), ('singleLR', 'respiratory', (0.7984654731457801, 0.20153452685421994, np.array([[3119,    3],
       [ 785,    3]]))), ('bagging LR', 'respiratory', (0.7982097186700767, 0.20153452685421994, np.array([[3118,    4],
       [ 785,    3]]))), ('singleSVM', 'respiratory', (0.7984654731457801, 0.20153452685421994, np.array([[3122,    0],
       [ 788,    0]]))), ('bagging SVM', 'respiratory', (0.7984654731457801, 0.20153452685421994, np.array([[3122,    0],
       [ 788,    0]]))), ('singleDT', 'respiratory', (0.6705882352941176, 0.20153452685421994, np.array([[2440,  682],
       [ 606,  182]]))), ('bagging DT', 'respiratory', (0.7936061381074169, 0.20153452685421994, np.array([[3086,   36],
       [ 771,   17]]))), ('singleNN', 'respiratory', (0.6946291560102302, 0.20153452685421994, np.array([[2550,  572],
       [ 622,  166]]))), ('bagging NN', 'respiratory', (0.7971867007672634, 0.20153452685421994, np.array([[3106,   16],
       [ 777,   11]]))), ('voting', 'skin', (0.6895140664961636, 0.6383631713554987, np.array([[ 399, 1015],
       [ 199, 2297]]))), ('boosting', 'skin', (0.7222506393861893, 0.6383631713554987, np.array([[ 499,  915],
       [ 171, 2325]]))), ('singleLR', 'skin', (0.6680306905370844, 0.6383631713554987, np.array([[ 557,  857],
       [ 441, 2055]]))), ('bagging LR', 'skin', (0.6687979539641944, 0.6383631713554987, np.array([[ 521,  893],
       [ 402, 2094]]))), ('singleSVM', 'skin', (0.6383631713554987, 0.6383631713554987, np.array([[   0, 1414],
       [   0, 2496]]))), ('bagging SVM', 'skin', (0.6383631713554987, 0.6383631713554987, np.array([[   0, 1414],
       [   0, 2496]]))), ('singleDT', 'skin', (0.6511508951406649, 0.6383631713554987, np.array([[ 737,  677],
       [ 687, 1809]]))), ('bagging DT', 'skin', (0.7419437340153453, 0.6383631713554987, np.array([[ 672,  742],
       [ 267, 2229]]))), ('singleNN', 'skin', (0.6271099744245524, 0.6383631713554987, np.array([[  93, 1321],
       [ 137, 2359]]))), ('bagging NN', 'skin', (0.6450127877237851, 0.6383631713554987, np.array([[ 256, 1158],
       [ 230, 2266]])))]

for i in l:
    print(i[0], i[1])
    print("recall: "+ str(round(i[2][2][1][1]/float( str(i[2][2][1][1] + i[2][2][1][0])), 5)))
    print("precision: " + str(round(i[2][2][1][1] / float(str(i[2][2][1][1] + i[2][2][0][1])), 5)))