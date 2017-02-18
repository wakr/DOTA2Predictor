import os

import dota2api
import json
import _thread

from flask import Flask, render_template, request, redirect, url_for, session
from flask_bootstrap import Bootstrap

from database.mongo import getDocuments
from settings import STEAMKEY, APPKEY
from machine_learning.match_learner import DOTA2Predictor
from machine_learning.match_parser import parseMatches, parseInputToFeatures
from data_mining.dota2data import main as mine

# Application configuration
api = dota2api.Initialise(STEAMKEY)
app = Flask(__name__)
app.secret_key = APPKEY
app.config['DEBUG'] = False
Bootstrap(app)
miningOnStartUp = True

def get_resource_as_string(name, charset='utf-8'):
    with app.open_resource(name) as f:
        return f.read().decode(charset)

app.jinja_env.globals['get_resource_as_string'] = get_resource_as_string
# End of application config

matches = []
heroes = []
predictor = None


def initialize_app():
    global matches, heroes, predictor, current_hero_count
    matches = getDocuments()
    heroes = api.get_heroes()['heroes']
    predictor = DOTA2Predictor(parseMatches(matches, heroes), heroes)


# Routes ##################################################

@app.route('/')
def main():
    count = len(matches)
    test_accuracy = 100 * round((1 - predictor.test_error_mean), 3)
    h = {}
    for hero in heroes:
        h[hero['id']] = hero
    top10Heroes = [(h[x[0]], x[1]) for i, x in enumerate(predictor.top10Heroes)]

    return render_template('index.html', count=count, heroes=heroes,
                           test_accuracy=test_accuracy, chart1=json.dumps(predictor.chart1),
                           top10=top10Heroes)

picks = []


@app.route('/results', methods=['POST', 'GET'])
def resultView():
    global picks
    if request.method == "POST":
        jsonData = request.get_json()
        picks = [int(p) for p in jsonData]
        return "Post received"
    else:
        print(picks)
        predict_vector = parseInputToFeatures(picks, heroes)
        prediction = [round(p, ndigits=2) for p in predictor.predict(predict_vector, True)[0]]
        winner = "Radiant" if predictor.predict(predict_vector)[0] else "Dire"
        return render_template('results.html', dire_pred=prediction[0] * 100, radiant_pred=prediction[1] * 100, winner=winner)


############################################################

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("Using port: " + str(port))
    initialize_app()
    _thread.start_new_thread(mine, ()) if miningOnStartUp else None
    app.run(host='0.0.0.0', port=port)
