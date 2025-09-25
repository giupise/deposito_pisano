# generate a function that can classify an iris dataset with scikit-learn
"""Iris classification utility using scikit-learn.

This module provides a small helper, :func:`classify_iris`, that trains a
``DecisionTreeClassifier`` and returns the accuracy on a provided test set.

It also exposes a simple CLI when executed as a script to quickly verify the
model performance on the built-in Iris dataset.

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def classify_iris(x_train, x_test, y_train, y_test):
    """Train a DecisionTreeClassifier on Iris data and evaluate accuracy.

    Args:
        x_train: 2D array-like of shape (n_train_samples, n_features)
            Training features.
        x_test: 2D array-like of shape (n_test_samples, n_features)
            Test features.
        y_train: 1D array-like of shape (n_train_samples,)
            Training labels.
        y_test: 1D array-like of shape (n_test_samples,)
            Test labels.

    Returns:
        float: Classification accuracy on the test set, in the range [0.0, 1.0].

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> data = load_iris()
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
        >>> round(classify_iris(X_train, X_test, y_train, y_test), 3) >= 0.8
        True
    """
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    # Simple manual test: load data, split, train, and print accuracy
    features, labels = load_iris(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    accuracy = classify_iris(train_features, test_features, train_labels, test_labels)
    print(f"DecisionTree accuracy on Iris test set: {accuracy:.3f}")
