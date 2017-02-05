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
        kf = KFold(n_splits=2)
        train, test = train_test_split(self.data, train_size=0.8)

        matrixedTrain = pd.get_dummies(train).as_matrix()
        trainX = matrixedTrain[:, :-1]  # take the end result away
        trainY = matrixedTrain[:, -1]  # # dire|radiant|direwon==1

        matrixedTest = pd.get_dummies(test).as_matrix()
        testX = matrixedTest[:, :-1]
        testY = matrixedTest[:, -1]

        logreg = LogisticRegression()
        logreg.fit(trainX, trainY)
        self.model = logreg

        print("Test accuracy: " + self.getScore(trainX, trainY))
        print("Train  accuracy" + self.getScore(testX, testY))

    def __init__(self, featureVector):
        self.data = featureVector
        self.model = None
        self.makeModel()

