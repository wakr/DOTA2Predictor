from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class DOTA2Predictor:


    def getTrainingRelation(self):
        return 0

    def predict(self, features):
        return self.model.predict(features)

    def getScore(self, X, Y):
        return self.model.score(X, Y)

    def makeModel(self):
        kf = KFold(n_splits=10)

        matrixedData = pd.get_dummies(self.data).as_matrix()
        X = matrixedData[:, :-1]  # take the end result away
        y = matrixedData[:, -1]  # # dire|radiant|direwon==1

        logreg = LogisticRegression(n_jobs=-1)

        train_error = []
        test_error = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logreg.fit(X_train, y_train)
            self.model = logreg

            train_error.append(1 - self.getScore(X_train, y_train))
            test_error.append(1 - self.getScore(X_test, y_test))

        print("Train and test errors per fold:")
        print(train_error)
        print(test_error)

    def __init__(self, featureVector):
        self.data = featureVector
        self.model = None
        self.makeModel()

