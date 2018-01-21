# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
import numpy as np

DATASET_PATH = 'datasets/user_modeling.data'
# dataset = np.loadtxt("datasets/user_modeling.data", delimiter=',', dtype=float)
dataset = []
with open(DATASET_PATH, 'r') as file:
	for line in file:
		dataset.append(line.strip().split(','))

for sample in dataset:
	for i in range(len(sample)-1):
		sample[i] = float(sample[i])

x = dataset[:, :4]
y = dataset[:, 5]
feature = preprocessing.scale(x)
model = KNeighborsClassifier()
model.fit(x, y)
print(model)
