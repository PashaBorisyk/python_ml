import pandas
import math
import matplotlib.pyplot as p
import numpy
from sklearn.svm import SVC as Model

data = pandas.read_csv("titanic.csv")
data_train = data[:int(0.8*len(data))]
data_test = data[int(0.8*len(data)):]

def cat_to_num(data):
    categories = pandas.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s=%s" % (data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)

def prepare_data(data):
    features = data.drop(["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    features["Age"] = data["Age"].fillna(-1)
    features["sqrt_Fare"] = numpy.sqrt(data["Fare"])
    features = features.join(cat_to_num(data['Sex']))
    features = features.join(cat_to_num(data["Embarked"]))
    return features

features = prepare_data(data_train)

model = Model()
model.fit(features, data_train["Survived"])
score = model.score(prepare_data(data_test), data_test["Survived"])
print(score)

p.plot(data_test.Fare, model.predict(prepare_data(data_test)), 'o')
p.show()
