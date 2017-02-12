from toolz.dicttoolz import keyfilter, assoc, dissoc
from toolz.itertoolz import groupby, concat

import pandas as pd


def decideSide(playerSlot):
    if playerSlot < 128:
        return 'R'
    else:
        return 'D'


def picksToHeroVector(picks, amountOfHeroes):
    featureVector = amountOfHeroes * [0]
    for heroID in picks:
        featureVector[heroID - 1] = 1
    return featureVector


def filterBadMatches(parsedMatches):
    isEnoughPlayers = lambda m: len(m['players']) == 10
    return list(filter(isEnoughPlayers, parsedMatches))


def filterFeatures(match):
    usableKeys = ['players', 'radiant_win', 'hero_id', 'player_slot']
    isUsable = lambda k: k in usableKeys
    toplvlFiltered = keyfilter(isUsable, match)

    filteredPlayers = []
    for player in toplvlFiltered['players']:
        side = decideSide(player['player_slot'])
        playerData = assoc(keyfilter(isUsable, player), 'team', side)
        playerData = dissoc(playerData, 'player_slot')
        filteredPlayers.append(playerData)
    toplvlFiltered['players'] = filteredPlayers

    return toplvlFiltered


def formFeatureMatrix(heroIDs, match):
    currentHeroAmount = len(heroIDs) + 1
    result = match['radiant_win']  # True if radiant won
    teams = groupby('team', match['players'])
    dire = teams['D']
    radiant = teams['R']

    # Dire is first, the Radiant

    matchVector = []
    for player in dire:
        matchVector.append(player['hero_id'])
    for player in radiant:
        matchVector.append(player['hero_id'])

    matchVector.append(result)

    finalVector = list(concat([(2 * currentHeroAmount) * [0], [0]]))
    for direPick in matchVector[:5]:
        normalizeDirePick = direPick - 1
        finalVector[normalizeDirePick] = 1
    for radiantPick in matchVector[5:10]:
        normalizeRadiantPick = currentHeroAmount + (radiantPick - 1)
        finalVector[normalizeRadiantPick] = 1

    if result:
        finalVector[-1] = 0 # dire lost aka radiant won
    else:
        finalVector[-1] = 1  # dire|radiant|direwon

    return finalVector


def parseMatches(matches, possibleHeroes):
    heroIDs = sorted(list(map(lambda h: h['id'], possibleHeroes)))
    parsedMatches = map(lambda match: filterFeatures(match), matches)
    parsedMatches = filterBadMatches(parsedMatches)
    featureMatrices = list(map(lambda m: formFeatureMatrix(heroIDs, m), parsedMatches))

    return pd.DataFrame(featureMatrices)

def parseInputToFeatures(data, possibleHeroes):
    heroCount = len(list(map(lambda h: h['id'], possibleHeroes))) + 1
    direSide = data[0:5]
    radiantSide = data[5:10]

    return picksToHeroVector(direSide, heroCount) + picksToHeroVector(radiantSide, heroCount)