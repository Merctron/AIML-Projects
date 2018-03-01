from __future__ import division
import numpy as np
import pandas as pd
import random
import sys
import math
import kmeans

events 	= pd.read_csv('./dataset/football-events/events.csv')
ginf   	= pd.read_csv('./dataset/football-events/ginf.csv')
players = pd.read_csv('./fifa-data.csv')

#events.info()
#ginf.info()

# K-Means Configuration
GINF_CONFIG = {
	'diff_cols': [9, 10, 11, 12, 13],
	'diff_labels': [
		'home_team_goals',
		'away_team_goals',
		'home_odds',
		'draw_odds',
		'away_odds'
	]
}

PLYR_CONFIG = {
	'diff_cols': [9, 10, 11, 12, 13],
	'diff_labels': [
		'home_team_goals',
		'away_team_goals',
		'home_odds',
		'draw_odds',
		'away_odds'
	]
}

# Gets team level statistics
def getTeamStats(league, team):
	league_games = ginf.loc[ginf['league'] == league]  
	home_games = league_games.loc[league_games['ht'] == team]
	away_games = league_games.loc[league_games['at'] == team]

	wins   = 0
	losses = 0
	draws  = 0
	h_wins = 0
	a_wins = 0

	for row in home_games.iterrows():
		if row[1][9] > row[1][10]:
			wins   += 1
			h_wins += 1
		elif row[1][9] == row[1][10]:
			draws += 1
		elif row[1][9] < row[1][10]:
			losses += 1

	for row in away_games.iterrows():
		if row[1][9] < row[1][10]:
			wins   += 1
			a_wins += 1
		elif row[1][9] == row[1][10]:
			draws += 1
		elif row[1][9] > row[1][10]:
			losses += 1
	print(team + ', Wins: {0}, Losses: {1}, Draws: {2}, Home win %: {3}, Away win %: {4}'.format(wins, losses, draws, 
		"{0:.2f}".format((h_wins / len(home_games.index)) * 100), 
		"{0:.2f}".format((a_wins / len(away_games.index)) * 100)))

# Gets league level statistics
def getLeagueStats(league):
	league_games = ginf.loc[ginf['league'] == league]
	teams = league_games.ht.unique()
	for team in teams:
		getTeamStats(league, team)

# Get players by position
def getPlayersByPos(players):
	player_lists = {}
	positions = ['rw','rm','rb','rwb','lw','lm','lb','lwb',
	'st','cf','cam','cm','cdm','cb','gk']

	for position in positions:
		player_lists[position] = players.loc[players['prefers_' + position] == True]

	for key, val, in player_lists.iteritems():
		print('Number of players in position, {0}: {1}'.format(key, 
			len(player_lists[key].index)))

	return player_lists




# getLeagueStats('E0')
# getLeagueStats('D1')
# getLeagueStats('F1')
# getLeagueStats('SP1')

# getTeamStats('E0', 'Manchester Utd')
# getTeamStats('E0', 'Liverpool')
# getTeamStats('E0', 'Chelsea')
# getTeamStats('E0', 'Manchester City')
# getTeamStats('E0', 'Arsenal')
# getTeamStats('E0', 'Tottenham')
# getTeamStats('D1', 'Bayern Munich')
# getTeamStats('F1', 'Paris Saint-Germain')
# getTeamStats('SP1', 'Real Madrid')
# getTeamStats('SP1', 'Barcelona')

# config = kmeans.KmeansConfig(GINF_CONFIG['diff_cols'], 
# 	GINF_CONFIG['diff_labels'], kmeans.DIFF_FUNC.EUCLIDEAN)
# kmeansObj = kmeans.Kmeans(config, 5, ginf)
# kmeansObj.cluster()

getPlayersByPos(players)