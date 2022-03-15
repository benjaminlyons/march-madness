import pandas as pd
import numpy as np

STATS = [ "W-L%", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

def main():
    game_logs_df = pd.read_csv("game_logs.csv")
    team_stats_df = pd.read_csv("team_stats.csv")

    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    print(team_stats_df.loc[team_stats_df['School'] == 'Duke'][STATS].to_numpy()[0])
    print(game_logs_df)

    with open("training_data.csv", "w") as f:
        for index, game in game_logs_df.iterrows():
            team1 = game["school1"]
            team2 = game["school2"]

            if team2 not in team_stats_df["School"].values:
                continue

            team1_stats = team_stats_df.loc[team_stats_df["School"] == team1][STATS].to_numpy()[0]
            team2_stats = team_stats_df.loc[team_stats_df["School"] == team2][STATS].to_numpy()[0]
            # stats = np.subtract(team1_stats, team2_stats).tolist()
            result = game["result"]

            # print(team1_stats)
            # print(team2_stats)
            f.write(",".join([str(x) for x in team1_stats]) + ",")
            f.write(",".join([str(x) for x in team2_stats]) + ",")
            # f.write(",".join([str(x) for x in stats]) + ",")
            f.write(str(result) + "\n")

if __name__ == "__main__":
    main()
