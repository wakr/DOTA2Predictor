from pymongo import MongoClient
from settings import DBUSER, DBPWD


def getURL(username, password):
    return "mongodb://%s:%s@ds061196.mlab.com:61196/dota2" % (username, password)


def getURL2(username, password):
    return "mongodb://%s:%s@ds155509.mlab.com:55509/dota2_large" % (username, password)

client = MongoClient(getURL2(DBUSER, DBPWD))
db = client.dota2_large

print(client)

def getDocuments():
    result = []
    try:
        result = list(db.matches.find({}, {"players.hero_id": 1, "players.player_slot": 1,
                                           "players.account_id": 1, "radiant_win": 1})
                                .limit(100))
    except Exception as e:
        print(e)
        return e
    return result


def insert_many(objects):
    db.matches.insert_many(objects, ordered=False)
