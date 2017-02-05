from pymongo import MongoClient
from settings import DBUSER, DBPWD


def getURL(username, password):
    return "mongodb://%s:%s@ds061196.mlab.com:61196/dota2" % (username, password)


client = MongoClient(getURL(DBUSER, DBPWD))
db = client.dota2


def getDocuments():
    return list(db.matches.find().limit(500))
