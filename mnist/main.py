import os
from time import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

############################# Loading data ###########################
os.chdir('../mnist')
mnist_data = pd.read_csv('data/train.csv')
print(mnist_data.shape)
X_data, y_data = mnist_data.iloc[:, 1:], mnist_data.iloc[:, 0]
X_data, y_data = X_data.values, y_data.values
X_set1, X_test, y_set1, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=5)
X_train, X_valid, y_train, y_valid = train_test_split(X_set1, y_set1, test_size=0.2, random_state=5)

print('dataset features shape: ', X_data.shape, ' dataset label shape: ', y_data.shape)
print(
    'shape of features used in training & validation: ', X_set1.shape,
    ' shape of label used in training & validation: ', y_set1.shape
)
print(
    'shape of features used in training: ', X_train.shape,
    ' shape label used in training: ', y_train.shape
)
print(
    'shape of features used in validation: ', X_valid.shape,
    ' shape label used in validation: ', y_valid.shape
)

############################# Pre-processing ##################################

from sklearn.decomposition import PCA

t0 = time()
pca = PCA(n_components=0.98).fit(X_train)
print('pca done in %0.3f', time() - t0)  # 20s
print('features kept', pca.n_components_)

X_train_pca = pca.transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

from sklearn.preprocessing import StandardScaler

scalar1 = StandardScaler()
X_train_scaled = scalar1.fit_transform(X_train.astype(np.float64))
X_valid_scaled = scalar1.transform(X_valid.astype(np.float64))
X_test_scaled = scalar1.transform(X_test.astype(np.float64))
scalar2 = StandardScaler()
X_train_scaled_pca = scalar2.fit_transform(X_train_pca.astype(np.float64))
X_valid_scaled_pca = scalar2.transform(X_valid_pca.astype(np.float64))
X_test_scaled_pca = scalar2.transform(X_test_pca.astype(np.float64))
# some_digit = 5
# img = Image.fromarray(X_train[some_digit].reshape(28, 28)).resize((250, 250)).show()
# y_train[some_digit]


############################ Linear Classifier ############################
from sklearn.linear_model import SGDClassifier

lin_clf = SGDClassifier(random_state=5, shuffle=True, tol=0.001)
t0 = time()
lin_clf.fit(X_train, y_train)
time_lin_clf = time() - t0
print('SGD classifier took {:0.3f}'.format(time_lin_clf))
y_pred_lin_clf = lin_clf.predict(X_valid)
score_lin_clf = accuracy_score(y_pred_lin_clf, y_valid)
print(score_lin_clf)

########################## SGD with scaled ################################
lin_clf_1 = SGDClassifier(random_state=5, shuffle=True, tol=0.001)
t0 = time()
lin_clf_1.fit(X_train_scaled, y_train)
time_lin_clf_scaled = time() - t0
print('scaled SGD classifier took {:0.3f}'.format(time_lin_clf_scaled))
y_pred_lin_clf_1 = lin_clf_1.predict(X_valid_scaled)
score_lin_clf_1 = accuracy_score(y_valid, y_pred_lin_clf_1)
print(score_lin_clf_1)

############################## SGD with pca ###############################
lin_clf_2 = SGDClassifier(random_state=5, shuffle=True, tol=0.001)
t0 = time()
lin_clf_2.fit(X_train_scaled_pca, y_train)
time_lin_clf_scaled_pca = time() - t0
print('scaled SGD classifier took {:0.3f}'.format(time_lin_clf_scaled_pca))
y_pred_lin_clf_2 = lin_clf_2.predict(X_valid_scaled_pca)
score_lin_clf_2 = accuracy_score(y_valid, y_pred_lin_clf_2)
print(score_lin_clf_2)


def show_image(image_array):
    Image.fromarray(image_array.astype(np.int32).reshape(28, 28)).resize((250, 250)).show()


X_valid_recovered = scalar1.inverse_transform(X_valid_scaled)
show_image(X_valid_recovered[0])

############################## Support Vector Machine #########################
from sklearn.svm import SVC

svm_clf = SVC()
t0 = time()
svm_clf.fit(X_train_scaled_pca, y_train)
time_svm_clf_scaled_pca = time() - t0
print('svm classifier took {:0.3f}'.format(time_svm_clf_scaled_pca))
y_pred_svm_clf = svm_clf.predict(X_valid_scaled_pca)
score_svm_clf = accuracy_score(y_valid, y_pred_svm_clf)
print(score_svm_clf)

########################## LinearSVC with pca ###########################
from sklearn.svm import LinearSVC

svm_clf_2 = LinearSVC()
t0 = time()
svm_clf_2.fit(X_train_scaled_pca, y_train)
time_svm_clf_2_scaled_pca = time() - t0
print('svm classifier took {:0.3f}'.format(time_svm_clf_2_scaled_pca))
y_pred_svm_clf_2 = svm_clf.predict(X_valid_scaled_pca)
score_svm_clf_2 = accuracy_score(y_valid, y_pred_svm_clf)
print(score_svm_clf_2)

############################ k_neighbors classifier #############################
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
t0 = time()
neigh.fit(X_train_pca, y_train)
time_neigh_clf_pca = time() - t0
print('k neighbor classifier took {:0.3f}'.format(time_neigh_clf_pca))  # 3569
y_pred_neigh = neigh.predict(X_valid_pca)
score_neigh_clf = accuracy_score(y_valid, y_pred_neigh)

########################## Decision Tree #####################################
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
t0 = time()
tree_clf.fit(X_train_pca, y_train)
time_tree_clf_pca = time() - t0
print('k neighbor classifier took {:0.3f}'.format(time_tree_clf_pca))
y_pred_tree_clf = tree_clf.predict(X_valid_pca)
score_tree_clf = accuracy_score(y_valid, y_pred_tree_clf)
print(score_tree_clf)

##################### GradientBoostingClassifier ############################

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()
t0 = time()
boost_clf.fit(X_train_pca, y_train)
time_boost_clf_pca = time() - t0
print(' GradBoost classifier took {:0.3f}'.format(time_boost_clf_pca))
y_pred_boost = boost_clf.predict(X_valid_pca)
print(accuracy_score(y_valid, y_pred_boost))
import _pickle

f = open('boost_clf.pkl', 'wb')
_pickle.dump(boost_clf, f)
f = open('boost_clf.pkl', 'rb')
clf = _pickle.load(f)
clf.score(X_valid_pca, y_valid)
##################### GradientBoostingClassifier ############################

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier()
t0 = time()
forest_clf.fit(X_train_pca, y_train)
time_forest_clf_pca = time() - t0
print(' Random Forest classifier took {:0.3f}'.format(time_forest_clf_pca))
