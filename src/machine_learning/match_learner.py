from toolz.dicttoolz import keyfilter, assoc, dissoc
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model

import matplotlib.pyplot as plt
import numpy as np


class DOTA2Predictor:

    def decideSide(self, playerSlot):
        if playerSlot < 128:
            return 'R'
        else:
            return 'D'


    def filterFeatures(self, match):
        usableKeys = ['players', 'radiant_win', 'hero_id', 'player_slot']
        isUsable = lambda k: k in usableKeys
        toplvlFiltered = keyfilter(isUsable, match)

        filteredPlayers = []
        for player in toplvlFiltered['players']:
            side = self.decideSide(player['player_slot'])
            playerData = assoc(keyfilter(isUsable, player), 'team', side)
            playerData = dissoc(playerData, 'player_slot')
            filteredPlayers.append(playerData)
        toplvlFiltered['players'] = filteredPlayers

        return toplvlFiltered

    def formFeatureMatrix(self, matches):
        pass

    def parseMatches(self, matches, possibleHeroes):
        heroIDs = sorted(list(map(lambda h: h['id'], possibleHeroes)))

        heroSelections = np.append(np.zeros(2 * len(heroIDs)), np.zeros(1)) # Radiant, Dire, Result
        parsedMatches = list(map(lambda match: self.filterFeatures(match), matches))
        print(parsedMatches[0])

        featureMatrix = self.formFeatureMatrix(parsedMatches)



    def __init__(self, matches, possibleHeroes):
        self.parseMatches(matches, possibleHeroes)

