import pandas
import math
import matplotlib.pyplot as pyplot
import numpy
from sklearn.neighbors import KNeighborsClassifier

mnist_data = pandas.read_csv("mnist_small.csv")
data_train = mnist_data[:int(0.8 * len(mnist_data))]
data_test = mnist_data[int(0.8 * len(mnist_data)):]

features_train = data_train.drop("label", axis=1)
features_test = data_test.drop("label",axis=1)

k_klassifier = KNeighborsClassifier(n_neighbors=10)

k_klassifier.fit(features_train, data_train["label"])
predictions = k_klassifier.predict_proba(features_test)

score = k_klassifier.score(features_test,data_test["label"])
print(score)