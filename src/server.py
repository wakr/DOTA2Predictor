import dota2api
import json

from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap

from database.mongo import getDocuments
from settings import STEAMKEY
from machine_learning.match_learner import DOTA2Predictor
from machine_learning.match_parser import parseMatches

# Application initializing
api = dota2api.Initialise(STEAMKEY)
app = Flask(__name__)
app.config['DEBUG'] = True
Bootstrap(app)

def get_resource_as_string(name, charset='utf-8'):
    with app.open_resource(name) as f:
        return f.read().decode(charset)

app.jinja_env.globals['get_resource_as_string'] = get_resource_as_string
# End of application init

matches = []
heroes = []
predictor = None

def initialize_app():
    global matches, heroes, predictor
    matches = getDocuments()
    heroes = api.get_heroes()['heroes']
    predictor = DOTA2Predictor(parseMatches(matches, heroes))

@app.route('/')
def main():
    count = len(matches)
    return render_template('index.html', count=count, heroes=heroes)


@app.route('/results', methods=['GET', 'POST'])
def resultView():
    if request.method == 'GET':
        return 'lol'
    else:
        jsonData = json.dumps(request.get_json())
        print(jsonData)
        return redirect(url_for('resultView'))


if __name__ == '__main__':
    initialize_app()
    app.run()
