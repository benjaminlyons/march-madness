import pandas as pd
import numpy as np
import random

STATS = [ "W-L%", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

def nn_training_data():
    game_logs_df = pd.read_csv("game_logs.csv")
    team_stats_df = pd.read_csv("team_stats.csv")

    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    print(team_stats_df.loc[team_stats_df['School'] == 'Duke'][STATS].to_numpy()[0])
    print(game_logs_df)

    with open("training_data.csv", "w") as f:
        for index, game in game_logs_df.iterrows():
            team1 = game["school1"]
            team2 = game["school2"]
            year = game["year"]

            if team2 not in team_stats_df["School"].values:
                continue

            team1_stats = team_stats_df.loc[team_stats_df["School"] == team1].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            try:
                team2_stats = team_stats_df.loc[team_stats_df["School"] == team2].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            except IndexError:
                continue
            # stats = np.subtract(team1_stats, team2_stats).tolist()
            result = game["result"]

            # print(team1_stats)
            # print(team2_stats)
            # randomize the order in which 
            if random.random() >= 0.5: 
                f.write(",".join([str(x) for x in team1_stats]) + ",")
                f.write(",".join([str(x) for x in team2_stats]) + ",")
                f.write(str(result) + "\n")
            else:
                f.write(",".join([str(x) for x in team2_stats]) + ",")
                f.write(",".join([str(x) for x in team1_stats]) + ",")
                f.write(str(1-result) + "\n")


def gp_training_data():
    game_logs_df = pd.read_csv("game_logs.csv")
    team_stats_df = pd.read_csv("team_stats.csv")

    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    print("Done with normalization")

    with open("spread_training_data.csv", "w") as f:
        for index, game in game_logs_df.iterrows():
            team1 = game["school1"]
            team2 = game["school2"]
            year = game["year"]

            if team2 not in team_stats_df["School"].values:
                continue

            if year != 2022:
                continue

            team1_stats = team_stats_df.loc[team_stats_df["School"] == team1].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            try:
                team2_stats = team_stats_df.loc[team_stats_df["School"] == team2].loc[team_stats_df["Year"]==year][STATS].to_numpy()[0]
            except IndexError:
                continue
            spread = game["score1"] - game["score2"]

            f.write(",".join([str(x) for x in team1_stats]) + ",")
            f.write(",".join([str(x) for x in team2_stats]) + ",")
            f.write(str(spread) + "\n")

def main():
    gp_training_data()
    nn_training_data()

if __name__ == "__main__":
    main()
