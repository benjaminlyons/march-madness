import pandas as pd
import torch
import numpy as np
from nn import Net

STATS = ["W-L%", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

def main():
    team_stats_df = pd.read_csv("team_stats.csv")
    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()

    model = torch.load("best_model.pth")['model'].cuda()
    
    while True:
        team1 = input("Please enter team1: ").strip()
        if not team1 or team1 == "quit":
            break
        team2 = input("Please enter team2: ").strip()

        team1_stats = team_stats_df.loc[team_stats_df["School"] == team1]
        team2_stats = team_stats_df.loc[team_stats_df["School"] == team2]
        
        print(f"Computing {team1} vs {team2}")

        team1_stats = team1_stats[STATS].to_numpy()
        team2_stats = team2_stats[STATS].to_numpy()

        x = np.concatenate((team1_stats, team2_stats), axis=1)
        x = torch.tensor(x, dtype=torch.float).cuda()
        output = model(x)

        result = output.item()

        if result >= 0.5:
            print(f"{team1} should win with probability {result}")
        else:
            print(f"{team2} should win with probability {1-result}")
        print()



if __name__ == "__main__":
    main()
