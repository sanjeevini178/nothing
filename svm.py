import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic "moons" dataset
X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot decision boundaries
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title(title)
    plt.show()

# SVM with a linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print("Linear Kernel SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("\nClassification Report (Linear Kernel):\n", classification_report(y_test, y_pred_linear))
plot_decision_boundary(svm_linear, X, y, "SVM with Linear Kernel")

# SVM with a polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)  # You can change the degree to experiment
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
print("Polynomial Kernel SVM Accuracy:", accuracy_score(y_test, y_pred_poly))
print("\nClassification Report (Polynomial Kernel):\n", classification_report(y_test, y_pred_poly))
plot_decision_boundary(svm_poly, X, y, "SVM with Polynomial Kernel")

# SVM with an RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print("RBF Kernel SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nClassification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))
plot_decision_boundary(svm_rbf, X, y, "SVM with RBF Kernel")
