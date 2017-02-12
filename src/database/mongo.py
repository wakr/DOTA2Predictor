from pymongo import MongoClient
from src.settings import DBUSER, DBPWD


def getURL(username, password):
    return "mongodb://%s:%s@ds061196.mlab.com:61196/dota2" % (username, password)


client = MongoClient(getURL(DBUSER, DBPWD))
db = client.dota2


def getDocuments():
    result = []
    try:
        result = list(db.matches.find({}, {"players.hero_id": 1, "players.player_slot": 1,
                                           "players.account_id": 1, "radiant_win": 1})
                                .limit(2000))
    except Exception as e:
        print(e)
    return result
