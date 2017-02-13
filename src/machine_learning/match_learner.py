from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
import numpy as np
import pandas as pd
import operator



class DOTA2Predictor:

    def __init__(self, featureVector, heroes):
        self.data = featureVector
        self.model = None
        self.train_error_mean = 0
        self.test_error_mean = 0

        self.hero_count = len(heroes)
        self.hero_data = heroes
        self.picked_heroes = None # concerns only the hero picks

        # analyzable features
        self.hero_distribution = []
        self.top10Heroes = []
        self.winLossRatios = []
        self.synergyPairs = [(4, 30), (2, 75), (40, 84), (8, 5), (23, 20), (104, 101), (18, 97), (41, 74), (30, 41)]

        self.chart1 = None
        self.chart2 = None
        self.chart3 = None
        self.makeModel()

    def predict(self, features, prob=False):
        f = np.array(features).reshape(1, -1)
        if prob:
            return self.model.predict_proba(f)
        else:
            return self.model.predict(f)

    def getScore(self, X, Y):
        return self.model.score(X, Y)

    def makeModel(self):
        kf = KFold(n_splits=10)

        matrixedData = self.data.as_matrix() # will be appended by top 10 played heroes
        X = matrixedData[:, :-1]  # take the end result away
        y = matrixedData[:, -1]  # # dire|radiant|direwon==1

        self.picked_heroes = matrixedData
        self.produceDataAnalysis()

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


    def produceHeroDistribution(self, hero_vector, hero_count_in_features):
        sum_vector = np.sum(hero_vector, axis=0)
        dire_side = sum_vector[:hero_count_in_features]
        radiant_side = sum_vector[hero_count_in_features:]

        hero_sum = np.add(dire_side, radiant_side)
        x = list(range(1, len(hero_sum) + 1))
        self.hero_distribution = hero_sum

        plt.bar(x, hero_sum, align="center")
        plt.xlabel("HeroID")
        plt.ylabel("Frequency")
        plt.title("Hero pick distribution")
        fig = plt.gcf()
        tooltip = plugins.MousePosition(fontsize=14)
        plugins.connect(fig, tooltip)
        self.chart1 = mpld3.fig_to_dict(fig)

    def formTop10MostPlayed(self):
        heroesWithIds = {}
        for i in range(len(self.hero_distribution)):
            ID = i+1
            heroesWithIds[ID] = self.hero_distribution[i]
        sorted_x = sorted(heroesWithIds.items(), key=operator.itemgetter(1), reverse=True)[:10]  # top ten
        self.top10Heroes = [x for x in sorted_x]

    def produceWinLoss(self):
        f = self.picked_heroes
        pass

    def produceDataAnalysis(self):
        hero_count_in_features = self.hero_count + 1
        hero_vector = self.picked_heroes[:, :-1]

        self.produceHeroDistribution(hero_vector, hero_count_in_features)
        self.formTop10MostPlayed()
        self.produceWinLoss()
