
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

dataset = np.loadtxt('watermelon_3.data', delimiter=',', dtype = 'string')

X = dataset[:, :6]
y = dataset[:, 6]
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

