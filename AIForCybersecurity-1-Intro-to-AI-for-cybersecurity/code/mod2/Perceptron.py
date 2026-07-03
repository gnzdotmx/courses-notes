from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X, y = make_classification(n_features=4, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = Perceptron(eta0=0.1, max_iter=1000)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))
