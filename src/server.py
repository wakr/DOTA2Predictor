import dota2api
import json

from flask import Flask, render_template, request, redirect, url_for, session
from flask_bootstrap import Bootstrap

from database.mongo import getDocuments
from settings import STEAMKEY, APPKEY
from machine_learning.match_learner import DOTA2Predictor
from machine_learning.match_parser import parseMatches, parseInputToFeatures

# Application configuration
api = dota2api.Initialise(STEAMKEY)
app = Flask(__name__)
app.secret_key = APPKEY
app.config['DEBUG'] = True
Bootstrap(app)

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
    top10Heroes = [(heroes[x[0]-1], x[1]) for i, x in enumerate(predictor.top10Heroes)]

    return render_template('index.html', count=count, heroes=heroes,
                           test_accuracy=test_accuracy, chart1=json.dumps(predictor.chart1),
                           top10=top10Heroes)


@app.route('/results', methods=['GET', 'POST'])
def resultView():
    if request.method == 'GET':
        picks = list(map(int, session['picks']))
        predict_vector = parseInputToFeatures(picks, heroes)
        prediction = list(map(lambda p: round(p, ndigits=2), predictor.predict(predict_vector, True)[0]))
        return render_template('results.html', dire_pred=prediction[0], radiant_pred=prediction[1])
    else:
        jsonData = request.get_json()
        session['picks'] = jsonData
        return redirect(url_for('resultView'), code=302)


############################################################

if __name__ == '__main__':
    initialize_app()
    app.run()
