# obtaining PCA for a dataset using numpy svd
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
X, y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

from PIL import Image
Image.fromarray(X_train[3].reshape((28, 28))).show()

# center the data first before applying PCA by your self
# X_centered = X_train - X_train.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered)
# c1 = Vt.T[:, 0]
# c2 = Vt.T[:, 1]
# projecting into first 2 PC
# W2 = Vt.T[:, :2]
# X2D = X_centered.dot(W2)

from sklearn.decomposition import PCA

pca = PCA(n_components=200)
# choose dimensions so that it adds up to sufficiently large portion of variance
X_reduced = pca.fit_transform(X_train)
#Image.fromarray(X2D[2].reshape(7, 7)).show()

# first component
print(pca.components_.T[:, 0])

# explained variance ratio od each component
# choose dimensions so that it adds up to sufficiently large portion of variance e.g. 95%
print(np.sum(pca.explained_variance_ratio_))

# finding right lower dimensions
pca = PCA() # no dim reduction
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 # so that we have >95
print(d)
 # or shotcut

pca = PCA(n_components=0.95) # float means variance
# choose dimensions so that it adds up to sufficiently large portion of variance
X_reduced = pca.fit_transform(X_train)
print(np.sum(pca.explained_variance_ratio_))
X_recovered = pca.inverse_transform(X_reduced)
print(X_recovered.shape)
Image.fromarray(X_recovered[2].reshape(28, 28)).show()

# incremental PCA
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

# virtual memoery
# X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
#
# batch_size = m // n_batches
# inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
# inc_pca.fit(X_mm)

# radomized pca
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)

# kernel pca
# from sklearn.decomposition import KernelPCA
#
# rbf_pca = KernelPCA(n_components = 150, kernel="rbf", gamma=0.04)
# X_reduced = rbf_pca.fit_transform(X_train)

#tuning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params_)

# reconstruction error
# rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
#                     fit_inverse_transform=True)
# X_reduced = rbf_pca.fit_transform(X)
# X_preimage = rbf_pca.inverse_transform(X_reduced)
# mean_squared_error(X, X_preimage)

# lle
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
