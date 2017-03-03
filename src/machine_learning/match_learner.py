from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

from machine_learning.pre_analyzer import *

import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
import numpy as np
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
        self.winLossRatios = {}
        self.winnlossSynergy = []
        self.counterSynergy = []
        self.synergyPairs = [(4, 30), (2, 75), (40, 84), (8, 5), (23, 20), (104, 101), (18, 97), (41, 74), (30, 41)]

        self.chart1 = None
        self.chart2 = None
        self.chart3 = None
        self.makeModel()


    def appendCrossFeatures(self, X, prediction=False):
        hero_count_in_features = self.hero_count + 1

        appendCross = lambda original, match_heroes: original \
                                               + get_heroSynergy_Diff(match_heroes, self.winnlossSynergy)\
                                               + get_counterSynergy(match_heroes, self.counterSynergy) \
                                               #+ get_distr_amount(match_heroes, self.top10Heroes) \
                                               #+ get_winlosses(matchHeroes, self.winLossRatios)

        # the data comes as a list
        if prediction:
            direSide = X[:hero_count_in_features]
            radiantSide = X[hero_count_in_features:]
            matchHeroes = get_selected_heroes(direSide) + get_selected_heroes(radiantSide)  # with real ID's
            enriched = appendCross(X, matchHeroes)
            return enriched
        else:
            # append win/loss
            X2 = []
            for match in X:
                mls = match.tolist()
                direSide = match[:hero_count_in_features]
                radiantSide = match[hero_count_in_features:]
                matchHeroes = get_selected_heroes(direSide) + get_selected_heroes(radiantSide) # with real ID's

                cfs = appendCross(mls, matchHeroes)

                X2.append(cfs)
            return np.matrix(X2)

    def predict(self, features, prob=False):
        crossF = self.appendCrossFeatures(features, True)
        f = np.array(crossF).reshape(1, -1)
        if prob:
            prediction = self.model.predict_proba(f)
            return prediction
        else:
            return self.model.predict(f)

    def getScore(self, X, Y):
        return self.model.score(X, Y)

    def makeModel(self):
        kf = KFold(n_splits=10)

        matrixedData = self.data.as_matrix() # will be appended by top 10 played heroes
        split_len = int(len(matrixedData) * 0.4) # reserving 40% of the data for analysis
        m1 = matrixedData[:split_len, :] # use for analyze
        m2 = matrixedData[split_len:, :] # use for model fitting
        self.picked_heroes = m1
        train, test = train_test_split(m2, train_size=0.8)

        tr_X = train[:, :-1]  # take the end result away
        tr_y = train[:, -1]  # dire|radiant|direwon==1
        ts_X = test[:, :-1]
        ts_y = test[:, -1]

        self.produceDataAnalysis()  # analyzers
        tr_X = self.appendCrossFeatures(tr_X)
        ts_X = self.appendCrossFeatures(ts_X)

        lr = LogisticRegression(n_jobs=-0.1)
        lr.fit(tr_X, tr_y)
        self.model = lr


        train_error = [1 - scr for scr in cross_val_score(lr, tr_X, tr_y, cv=kf)]
        test_error = [1 - scr for scr in cross_val_score(lr, ts_X, ts_y, cv=kf)]
        print("Train and test errors per fold:")
        print(train_error)
        print(test_error)
        self.train_error_mean = float(np.mean(train_error))
        self.test_error_mean = float(np.mean(test_error))
        print("Train error mean: " + str(self.train_error_mean))
        print("Test error mean: " + str(self.test_error_mean))

        print("\nConfusion matrix (Radiant=0, Dire=1):")
        print(confusion_matrix(ts_y, self.model.predict(ts_X)))



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

    def formWinLoss(self):
        hero_count_in_features = self.hero_count + 1
        h = {x: {"win": 0, "loss": 0} for (i, x) in enumerate(range(1, hero_count_in_features + 1))}
        wl = {}
        for match in self.picked_heroes:
            result = match[-1]
            direSide = match[:hero_count_in_features]
            radiantSide = match[hero_count_in_features:-1]  # take the result away

            for ID, b in enumerate(direSide):
                if b:  # if picked
                    if result:
                        h[ID + 1]["win"] += 1
                    else:  # dire lost == 0
                        h[ID + 1]["loss"] += 1
            for ID, b in enumerate(radiantSide):
                if b:  # if picked
                    if not result:
                        h[ID + 1]["win"] += 1
                    else:
                        h[ID + 1]["loss"] += 1

        for ID in h.keys():
            val = h[ID]
            try:
                wl_ratio = val["win"] / (val["win"] + val["loss"])
                wl[ID] = round(wl_ratio, 3)
            except Exception:
                wl[ID] = 0

        self.winLossRatios = wl

    def formPairwiseSynergy(self):
        hero_count_in_features = self.hero_count + 1
        winMatrix = np.zeros((hero_count_in_features, hero_count_in_features), dtype=np.int)
        lossMatrix = np.zeros((hero_count_in_features, hero_count_in_features), dtype=np.int)
        ratioMatrix = np.zeros((hero_count_in_features, hero_count_in_features))

        for match in self.picked_heroes:
            result = match[-1]
            direSide = match[:hero_count_in_features]
            radiantSide = match[hero_count_in_features:-1]

            dt = []  # contains heroIDs with -1 off
            rt = []
            for ID, b in enumerate(direSide):
                if b:
                    dt.append(ID)
            for ID, b in enumerate(radiantSide):
                if b:
                    rt.append(ID)

            for ID in dt:
                for pairID in dt:
                    if result: # dire won
                        winMatrix[ID, pairID] += 1
                        winMatrix[pairID, ID] += 1
                    else:
                        lossMatrix[ID, pairID] += 1
                        lossMatrix[pairID, ID] += 1

            for ID in rt:
                for pairID in rt:
                    if not result:  # radiant won
                        winMatrix[ID, pairID] += 1
                        winMatrix[pairID, ID] += 1
                    else:
                        lossMatrix[ID, pairID] += 1
                        lossMatrix[pairID, ID] += 1

        for i in range(0, hero_count_in_features):
            for j in range(0, hero_count_in_features):
                if i == j: # diagonal should be 0
                    ratioMatrix[i, j] = 0
                else:
                    totalSum = (winMatrix[i, j] + lossMatrix[i, j])
                    if not totalSum:
                        ratioMatrix[i, j] = 0
                    else:
                        pass
                        ratioMatrix[i, j] = round(winMatrix[i, j] / totalSum, 4)


        self.winnlossSynergy = ratioMatrix
        #plt.figure(4)
        #plt.xlabel("HeroID")
        #plt.ylabel("HeroID")
        #plt.title("Pairwise synergies")
        #plt.imshow(ratioMatrix, origin="low", interpolation="none", aspect="auto", cmap="inferno")
        #plt.colorbar()
        #plt.show()


    def formCounterSynergy(self):
        hero_count_in_features = self.hero_count + 1
        winMatrix = np.zeros((hero_count_in_features, hero_count_in_features), dtype=np.int)
        lossMatrix = np.zeros((hero_count_in_features, hero_count_in_features), dtype=np.int)
        ratioMatrix = np.zeros((hero_count_in_features, hero_count_in_features))

        for match in self.picked_heroes:
            result = match[-1]
            direSide = match[:hero_count_in_features]
            radiantSide = match[hero_count_in_features:-1]

            dt = []  # contains heroIDs with -1 off
            rt = []
            for ID, b in enumerate(direSide):
                if b:
                    dt.append(ID)
            for ID, b in enumerate(radiantSide):
                if b:
                    rt.append(ID)

            for ID in dt:
                for pairID in rt:
                    if result: # dire won
                        winMatrix[ID, pairID] += 1
                        lossMatrix[pairID, ID] += 1
                    else:
                        lossMatrix[ID, pairID] += 1
                        winMatrix[pairID, ID] += 1

            for ID in rt:
                for pairID in dt:
                    if not result:  # radiant won
                        winMatrix[ID, pairID] += 1
                        lossMatrix[pairID, ID] += 1
                    else:
                        lossMatrix[ID, pairID] += 1
                        winMatrix[pairID, ID] += 1

        for i in range(0, hero_count_in_features):
            for j in range(0, hero_count_in_features):
                if i == j: # diagonal should be 0
                    ratioMatrix[i, j] = 0
                else:
                    totalSum = (winMatrix[i, j] + lossMatrix[i, j])
                    if not totalSum:
                        ratioMatrix[i, j] = 0
                    else:
                        ratioMatrix[i, j] = round(winMatrix[i, j] / totalSum, 4)


        self.counterSynergy = ratioMatrix
        #plt.figure(5)
        #plt.xlabel("HeroID")
        #plt.ylabel("HeroID")
        #plt.title("Counter synergies")
        #plt.imshow(ratioMatrix, origin="low", interpolation="none", aspect="auto", cmap="inferno")
        #plt.colorbar()
        #plt.show()

    def produceDataAnalysis(self):
            hero_count_in_features = self.hero_count + 1
            hero_vector = self.picked_heroes[:, :-1]

            self.produceHeroDistribution(hero_vector, hero_count_in_features)
            self.formTop10MostPlayed()
            self.formWinLoss()
            self.formPairwiseSynergy()
            self.formCounterSynergy()
