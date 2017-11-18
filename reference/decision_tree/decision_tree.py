from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# visualise tree
from sklearn.tree import export_graphviz

export_graphviz(tree_clf,
                out_file="iris_tree.dot",  # output file
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True)

# dot -Tng iris_tree.dot -o iris_tree.png

tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
