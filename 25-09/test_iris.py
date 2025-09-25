from iris import classify_iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_classify_iris_accuracy_reasonable():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    acc = classify_iris(X_train, X_test, y_train, y_test)
    # DecisionTree on Iris should usually be above 0.8 with this split
    assert acc >= 0.8

