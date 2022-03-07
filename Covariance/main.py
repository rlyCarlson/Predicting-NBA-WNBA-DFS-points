
# Make sure calls to NBA_API does not timeout
headers  = {
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Libraries
import requests
import statistics
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.endpoints import playercareerstats
import random
from numpy import linalg
from bs4 import BeautifulSoup
import unicodedata
from tabulate import tabulate

DFS_POINTS = ['FGM', 'FG3M', 'REB', 'AST', 'BLK', 'STL', 'TO']
DFS_POINT_VALUES = [1, 0.5, 1.25, 1.5, 2, 2, -0.5]

def find_stats(simulations, gamePlayers):
    means = []
    stds = []
    for player in range(len(gamePlayers)):
        numbers = np.array([])
        for sim in range(len(simulations)):
            numbers = np.append(numbers, simulations[sim][player])
        means.append(np.mean(numbers))
        stds.append(np.std(numbers))
    return means, stds


def check_valid(team_input, abbreviations, nicknames):
    # Check if team abbreviation inputed is valid
    while (not (team_input in abbreviations)):
        team_input = input("Error team abbreviation not correct (enter 'h' for help) ")
        if (team_input == 'h'):
            print("Abbreviations:")
            for i in range(len(abbreviations)):
                print(nicknames[i], abbreviations[i])
    return team_input

def add_matrices(matrices):
    # Go through all the elements in each row of matrices and add
    prediction = []
    num_sections_double_digits = [0 for _ in range(len(matrices[0]))]
    # e = row (player)
    for e in range(len(matrices[0])):
        temp = 0
        # k = point type
        for k in range(len(matrices)):
            # Double-double or triple-double
            if(k == 0 or k == 2 or k == 3 or k == 4 or k == 5):
                if(abs(matrices[k][e]) >= 10):
                    num_sections_double_digits[e] += 1
            if(k == len(matrices) - 1):
                temp -= matrices[k][e]
            else:
                temp += matrices[k][e]
        prediction.append(temp)
    for x in range(len(num_sections_double_digits)):
        if(x >= 2):
            prediction[x] += 1.5
        if(x >= 3):
            prediction[x] += 3
    return prediction

def add(mu, CZ, point_number):
    # Add each element in matrices
    final = []
    for e in range(len(mu)):
        temp = float(mu[e]) + float(CZ[e])
        # print(temp)
        temp *= DFS_POINT_VALUES[point_number]
        # print(temp)
        final.append(temp)
    return final

def get_matrix(data):
    # Go through data and get values for each fantasy point
    summary = []
    summary.append(float(data['FG']))
    summary.append(float(data['3P']))
    summary.append(float(data['ORB']) + float(data['DRB']))
    summary.append(float(data['AST']))
    summary.append(float(data['STL']))
    summary.append(float(data['BLK']))
    summary.append(float(data['TOV']))
    return summary

def find_player(player, df):
    # Go through data frame and compare names ignoring diacritics
    for i in df.index:
        if(unicodedata.normalize('NFKD', df.iloc[i]['Player']).encode('ascii', 'ignore').decode("utf-8") == player):
            return i
    return -1

def get_BBallRef_data(year, game_players):
    # Webscrape
    url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
    r = requests.get(url)
    r_html = r.text
    soup = BeautifulSoup(r_html, 'html.parser')

    # Style the data
    table = soup.find_all(class_="full_table")
    head = soup.find(class_="thead")
    column_names_raw = [head.text for item in head][0]
    column_names_polished = column_names_raw.replace("\n", ",").split(",")[2: -1]

    # Build data frame
    p = []
    for i in range(len(table)):
        player = []
        for td in table[i].find_all("td"):
            player.append(td.text)
        p.append(player)
    df = pd.DataFrame(p, columns=column_names_polished)
    df.index = [i for i in range(0, len(table))]

    summary = [[0 for x in range(len(DFS_POINTS))] for y in range(len(game_players))]
    j = 0
    # Find summary data for each player
    for player_i in game_players:
        player_i_name = players.find_player_by_id(str(int(player_i)))['full_name']
        index = find_player(player_i_name, df)
        if(index != -1):
            summary[j] = get_matrix(df.loc[index,:])
        else:
            print("ERROR: name not found " + player_i_name)
        j += 1
    return summary

def create_covariance_matrix(point_type, gamePlayers, game_ids):
    N = len(gamePlayers)
    covariance_arr= [[1 for _ in range(N)] for _ in range(N)]
    # Take two players and find covariance between them
    for player1Index in range(0, N-1):
        for player2Index in range(player1Index + 1, N):
            means = np.array([])
            # Go through each game between two teams
            for game_num in range(0, len(game_ids)):
                player_stat_data_df = psd_df_collection[game_ids[game_num]]
                # See if both players are playing in same game
                if((int(gamePlayers[player1Index]) in player_stat_data_df['PLAYER_ID'].values) and (int(gamePlayers[player2Index]) in player_stat_data_df['PLAYER_ID'].values)):
                    #Find player1 and player2 data in this game
                    row1 = player_stat_data_df[player_stat_data_df['PLAYER_ID'] == gamePlayers[player1Index]].index.to_numpy()
                    row2 = player_stat_data_df[player_stat_data_df['PLAYER_ID'] == gamePlayers[player2Index]].index.to_numpy()
                    num1 = player_stat_data_df.at[row1[0], point_type]
                    num2 = player_stat_data_df.at[row2[0], point_type]
                    mean = statistics.mean([num1, num2])
                    means = np.append(means, mean)
            # Find variance
            if(len(means) > 1):
                variance = statistics.variance(means)
            else:
                variance = 0
            # Make sure covariance matrix is symmetric
            covariance_arr[player1Index][player2Index] = variance
            covariance_arr[player2Index][player1Index] = variance
    return np.array(covariance_arr)


#Get teams
nba_teams = teams.get_teams()
team_df = pd.DataFrame(nba_teams)
team_abbreviations = team_df['abbreviation'].to_numpy()
team_nicknames = team_df['nickname'].to_numpy()
team_ids = pd.DataFrame(team_df[['id','abbreviation']])

# Input teams
teamAAbr = input("Enter team A abbreviation: ")
# Check if valid input
teamAAbr = check_valid(teamAAbr, team_abbreviations, team_nicknames)
print("team A accepted")
teamBAbr = input("Enter team B abbreviation: ")
teamBAbr = check_valid(teamBAbr, team_abbreviations, team_nicknames)
print("team B accepted")
# teamAAbr = 'GSW' # Warrior
# teamBAbr = 'LAL' # Lakers
teamA_id = 0
teamB_id = 0
# Find team ids
for i in range(0, len(team_ids)):
    if(team_ids.at[i, 'abbreviation'] == teamAAbr):
        teamA_id = team_ids.at[i, 'id']
    if(team_ids.at[i, 'abbreviation'] == teamBAbr):
        teamB_id = team_ids.at[i, 'id']

#Get games where teamA and teamB played
print("\tGathering games between {} and {}...".format(teamAAbr, teamBAbr))
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable = '2020-21', league_id_nullable='00', season_type_nullable='Regular Season', headers=headers, timeout=100)
games = gamefinder.get_data_frames()[0]
games_df = pd.DataFrame(games)
matchups_ids = pd.DataFrame(games_df[['GAME_ID', 'MATCHUP']])
game_ids = np.array([])
# See if teamA played teamB
for i in matchups_ids.iterrows():
    game_i_id = i[1][0]
    game_i_matchup = i[1][1]
    teamA = game_i_matchup[0:3]
    teamB = game_i_matchup[-3:-1]+game_i_matchup[-1]
    if((teamA == teamAAbr and teamB == teamBAbr) or (teamA == teamBAbr and teamB == teamAAbr)):
        game_ids = np.append(game_ids, game_i_id)

# Finding player id's
print("\tGathering player information...")
gamePlayers = np.array([])
psd_df_collection = {}
for g in game_ids:
    psd = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=g, headers=headers, timeout=100)
    psd_df = psd.get_data_frames()[0]
    gamePlayers = np.append(gamePlayers, psd_df['PLAYER_ID'].to_list())
    psd_df_collection[g] = pd.DataFrame(psd_df)
gamePlayers = np.unique(gamePlayers)
# Get summary statistics for each player
summaries = get_BBallRef_data("2021", gamePlayers)

# lebron_id = 2544
# curry_id = 201939
# gamePlayers = np.append(gamePlayers, lebron_id)
# gamePlayers = np.append(gamePlayers, curry_id)
# print("players", gamePlayers)

print("\tCreating Covariance Matrices...")
N = len(gamePlayers)
covariances = np.array([])
t = []
for i in range(len(DFS_POINTS)):
    temp = create_covariance_matrix(DFS_POINTS[i], gamePlayers, game_ids)
    covariances = np.append(covariances, temp)
    for index, n in np.ndenumerate(temp):
        if (pd.isna(n)):
            temp[index[0]][index[1]] = 0
    t.append(temp)

print("\tPerforming Cholesky decomposition...")
# Perform spectral decomposition to make matrix positve definite
decomposed_matrices = []
for i in range(len(t)):
    matrix = t[i]
    u, Q = np.linalg.eig(matrix)
    D = np.diag(u).real
    Q_T = np.linalg.inv(Q)
    # Perform Spectral Decomposition
    for j in range(0, len(D)):
        if(D[j][j] <= 0.001):
            D[j][j] = 0.001


    # Reassemble matrix
    new_matrix = Q @ D @ Q_T
    new_matrix = new_matrix.real
    # Perform Cholesky decomposition on covariance matrix
    decomposed_matrix = np.linalg.cholesky(new_matrix)
    decomposed_matrices.append(decomposed_matrix)

# Do 1000 simulations to find variance
NUM_SIMULATIONS = 1000
print("\tPerforming {} simulations".format(NUM_SIMULATIONS))
predictions = []
for simulation in range(NUM_SIMULATIONS):
    final_matrices = []
    for m in range(len(decomposed_matrices)):
        # vector containing random numbers between 0 and 1
        n_vector = np.array([[random.uniform(0, 1)] for k in range(0, len(gamePlayers))])
        # summaries: 32 rows 7 columns
        means_dfs_i = [row[m] for row in summaries]
        # Combine matrices to get expected number for each type of fantasy point
        mu_Cz = add(means_dfs_i, (decomposed_matrices[m] @ n_vector), m)
        final_matrices.append(mu_Cz)
    # Add all the expected number of fantasy points into one matrix
    prediction = add_matrices(final_matrices)
    predictions.append(prediction)

means, stds = find_stats(predictions, gamePlayers)

print("PREDICTED FANTASY POINT EARNING")
output = []
for element in range(len(gamePlayers)):
    o = ["{}".format(players.find_player_by_id(str(int(gamePlayers[element])))['full_name']), "\t\t{:.3f}".format(means[element] - stds[element]), "\t{:.3f}".format(means[element]), "\t{:.3f}".format(means[element] + stds[element])]
    output.append(o)

print(tabulate(output, headers=["NAME", "-1 STD", "Mean", "+1 STD"]))