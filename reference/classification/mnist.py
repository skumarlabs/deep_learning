import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# download mnist
mnist = fetch_mldata("MNIST original")

X, y = mnist['data'], mnist['target']
print("feature dataset shape", X.shape)
print("target dataset shape", y.shape)

# check one digit
Image.fromarray(X[37000].reshape(28, 28)).show()
plt.imshow(X[37000].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation="nearest")

# verify it
print(y[37000])
digit_6 = X[37000]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffle the order
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_6 = (y_train == 6)  # True for all 6s, false for all other digit
y_test_6 = (y_test == 6)

sgd_clf = SGDClassifier(random_state=5)
sgd_clf.fit(X_train, y_train_6)

sgd_clf.predict([digit_6])

# cross validation
skfolds = StratifiedKFold(n_splits=5, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_6):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_6[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_6[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))  # accuracy

cross_val_score(sgd_clf, X_train, y_train_6, cv=6, scoring="accuracy")


class Never6Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never6_clf = Never6Classifier()
cross_val_score(never6_clf, X_train, y_train_6, cv=6, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_6, cv=6)

# rows correspond to correct class(negative class first), column to predicted class
# A prefect classifier will have off diagonal values 0.
conf_mat = confusion_matrix(y_train_6, y_train_pred)
precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])  # TP/(TP+FP)
recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])  # TP/(TP+FN)

print(precision)
print(recall)

print(precision_score(y_train_6, y_train_pred))
print(recall_score(y_train_6, y_train_pred))
print(f1_score(y_train_6, y_train_pred))

# Raising threshold increases precision but reduces recall
y_scores = sgd_clf.decision_function([digit_6])
print(y_scores)
threshold = 0
y_6_digit_pred = (y_scores > threshold)
print(y_6_digit_pred)

threshold = 400000
y_6_digit_pred = (y_scores > threshold)
print(y_6_digit_pred)

# deciding threshold
y_scores = cross_val_predict(sgd_clf, X_train, y_train_6, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_6, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

y_train_pred_90 = (y_scores > 70000)
print(precision_score(y_train_6, y_train_pred_90))
print(recall_score(y_train_6, y_train_pred_90))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_6, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_train_6, y_scores))

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_6, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_6, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print(roc_auc_score(y_train_6, y_scores_forest))

# multi class classification
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([digit_6]))

some_digit_scores = sgd_clf.decision_function([digit_6])
print(some_digit_scores)
print(np.argmax(some_digit_scores))

print(sgd_clf.classes_)

from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([digit_6]))
print(len(ovo_clf.estimators_))

#  random forest
forest_clf.fit(X_train, y_train)
forest_clf.predict([digit_6])
forest_clf.predict_proba([digit_6])

# cross valid
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# afteree scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([digit_6])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# Multi-output Multi-class classification
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
some_digit = X_test_mod[500].astype(np.uint32)
clean_digit = knn_clf.predict([some_digit])
Image.fromarray(clean_digit.reshape(28, 28)).show()
Image.fromarray(some_digit.reshape(28, 28)).show()
print(type(some_digit[0]))
