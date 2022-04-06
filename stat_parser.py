import pandas as pd
import numpy as np
import random
from compute_homefield_adv import compute_homefield_advs

STATS = [ "W-L%", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

def nn_training_data():
    game_logs_df = pd.read_csv("data/all_game_logs.csv")
    team_stats_df = pd.read_csv("data/all_team_stats.csv")

    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    print(team_stats_df.loc[team_stats_df['School'] == 'Duke'][STATS].to_numpy()[0])
    print(game_logs_df)
    homefield_advs, _ = compute_homefield_advs()

    with open("training_data.csv", "w") as f:
        for index, game in game_logs_df.iterrows():
            team1 = game["school1"]
            team2 = game["school2"]
            year = game["year"]

            if team2 not in team_stats_df["Link Name"].values:
                print("Team2 not found", team2)
                continue

            team1_stats = team_stats_df.loc[team_stats_df["School"] == team1].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            try:
                team2_stats = team_stats_df.loc[team_stats_df["Link Name"] == team2].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            except IndexError:
                continue
            # stats = np.subtract(team1_stats, team2_stats).tolist()
            result = game["result"]
            location = game["location"]

            # print(team1_stats)
            # print(team2_stats)
            home_adv = 0
            if location == 1:
                home_adv = homefield_advs[team1][0]
            elif location == -1:
                team2_name = team_stats_df.loc[team_stats_df["Link Name"] == team2].loc[team_stats_df["Year"] == year]["School"].item()
                home_adv = -homefield_advs[team2_name][0]

            spread = game["score1"] - game["score2"]
            spread += home_adv

            # use home_adjusted spread to determine who should have won the game
            if spread > 0:
                result = 1
            else:
                result = 0

            # f.write(str(location) + ",")
            f.write(",".join([str(x) for x in team1_stats]) + ",")
            f.write(",".join([str(x) for x in team2_stats]) + ",")
            f.write(str(result) + "\n")

def gp_training_data():
    game_logs_df = pd.read_csv("game_logs.csv")
    team_stats_df = pd.read_csv("team_stats.csv")

    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    print("Done with normalization")
    # print(team_stats_df)
    # print(team_stats_df["School"].values)

    homefield_advs, _ = compute_homefield_advs()

    with open("spread_training_data.csv", "w") as f:
        for index, game in game_logs_df.iterrows():
            team1 = game["school1"]
            team2 = game["school2"]
            year = game["year"]
            loc = game["location"]

            if team2 not in team_stats_df["Link Name"].values:
                print("Team2 not found:", team2)
                continue

            if year != 2022:
                continue

            team1_stats = team_stats_df.loc[team_stats_df["School"] == team1].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            try:
                team2_stats = team_stats_df.loc[team_stats_df["Link Name"] == team2].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            except IndexError:
                print("Index error")
                continue

            spread = game["score1"] - game["score2"]

            if loc == 1:
                spread -= homefield_advs[team1][0]
            elif loc == -1:
                team2_name = team_stats_df.loc[team_stats_df["Link Name"] == team2].loc[team_stats_df["Year"] == year]["School"].item()
                spread += homefield_advs[team2_name][0]

            f.write(",".join([str(x) for x in team1_stats]) + ",")
            f.write(",".join([str(x) for x in team2_stats]) + ",")
            f.write(str(spread) + "\n")

def main():
    # gp_training_data()
    nn_training_data()

if __name__ == "__main__":
    main()
