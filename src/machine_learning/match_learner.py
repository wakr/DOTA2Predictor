from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class DOTA2Predictor:

    def __init__(self, featureVector, hero_count):
        self.data = featureVector
        self.model = None
        self.train_error_mean = 0
        self.test_error_mean = 0
        self.hero_count = hero_count
        self.makeModel()

    def predict(self, features, prob = False):
        f = np.array(features).reshape(1, -1)
        if prob:
            return self.model.predict_proba(f)
        else:
            return self.model.predict(f)

    def getScore(self, X, Y):
        return self.model.score(X, Y)

    def makeModel(self):
        kf = KFold(n_splits=10)

        matrixedData = self.data.as_matrix()
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
        self.train_error_mean = float(np.mean(train_error))
        self.test_error_mean = float(np.mean(test_error))
        print("Train error mean: " + str(self.train_error_mean))
        print("Test error mean: " + str(self.test_error_mean))

        self.produceDataAnalysis(X)

    def produceDataAnalysis(self, X):
        hero_count_in_features = self.hero_count + 1
        hero_vector = X[:, :hero_count_in_features * 2]
        sum_vector = np.sum(hero_vector, axis=0)
        dire_side = sum_vector[:hero_count_in_features]
        radiant_side = sum_vector[hero_count_in_features:]

        hero_sum = np.add(dire_side, radiant_side)
        x = list(range(1, len(hero_sum) + 1))
        plt.bar(x, hero_sum, align="center")
        plt.show()