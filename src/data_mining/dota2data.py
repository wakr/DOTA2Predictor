from toolz import dicttoolz, assoc, dissoc, take
from database.mongo import insert_many, db
from settings import STEAMKEY

import dota2api
import json


api = dota2api.Initialise(STEAMKEY)
heroes = api.get_heroes()['heroes']

def get_matches_with_players(response):
    return list(filter(lambda match: 'players' in match, response))


def get_only_top_tier_games(response):
    return list(filter(lambda match: match['game_mode'] == 22, response)) # ranked all pick


def enrich_player_data(response):
    hero_dict = {}

    for hero in heroes:
        hero_dict[hero['id']] = hero

    for game in response:
        players = game['players']

        for player in players:
            player_hero_id = player['hero_id']
            player['hero'] = hero_dict[player_hero_id]
            del(player['hero_id'])

    return response


def get_live_match_data():
    played_games = api.get_top_live_games()['game_list']
    games_with_players = get_matches_with_players(played_games)
    correct_game_modes = get_only_top_tier_games(games_with_players)
    enriched_players = enrich_player_data(correct_game_modes)
    return enriched_players


def get_live_players():
    unique_ids = set()
    played_games = api.get_top_live_games()['game_list']
    games_with_players = get_matches_with_players(played_games)

    for match in games_with_players:
        players = match['players']
        for player in players:
            unique_ids.add(player['account_id'])

    return list(unique_ids)


def get_match_details(match_id):
    return api.get_match_details(match_id)


def parse_match_ids_from_player_data(player_data):
    return list(map(lambda x: x['match_id'], player_data['matches']))


def minify_data(collected):
    minified = []
    for c in collected:
        rd_win = c['radiant_win']
        match_id = c['match_id']
        player_data = []
        for p in c['players']:
            h_id = p['hero_id']
            p_slot = p['player_slot']
            acc_id = p['account_id']
            player_data.append({"hero_id": h_id, "player_slot": p_slot, "account_id": acc_id})

        if len(player_data) == 10:
            minified.append({"radiant_win": rd_win, "match_id": match_id, "players": player_data})
    return minified

def main():

    print("mining started")

    collective_match_data = []
    try:
        top_players = get_live_players()

        for player_id in top_players:
            try:
                matches = api.get_match_history(player_id)
            except:
                continue
            match_ids = parse_match_ids_from_player_data(matches)

            for match_id in list(take(3, match_ids)):
                match_details = api.get_match_details(match_id)
                collective_match_data.append(match_details)

        minified = minify_data(collective_match_data)
        insert_many(minified)
    except Exception as e:
        print(e)

    print("miner added " + str(len(collective_match_data)) + " new matches")





if __name__ == '__main__':
    main()

