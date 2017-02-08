import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DBUSER = os.environ.get("DBUSER")
DBPWD = os.environ.get("DBPWD")
STEAMKEY = os.environ.get("STEAMKEY")
APPKEY = os.environ.get("APPKEY")