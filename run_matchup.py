import pandas as pd
import torch
import numpy as np
import random
import sys
from nn import Net

STATS = ["W-L%", "SRS", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

def compute_expected_result(team1, team2, team_stats_df, model):
    team1_stats = team_stats_df.loc[team_stats_df["School"] == team1]
    team2_stats = team_stats_df.loc[team_stats_df["School"] == team2]

    team1_stats = team1_stats[STATS].to_numpy()
    team2_stats = team2_stats[STATS].to_numpy()

    x = np.concatenate((team1_stats, team2_stats), axis=1)
    x = torch.tensor(x, dtype=torch.float).cuda()

    output = model(x)

    result = output.item()

    return result

# reads bracket from filename, then computes results
def read_bracket(filename):
    bracket = []
    with open(filename, 'r') as f:
        for team in f.readlines():
            bracket.append(team.strip())
    return bracket

def compute_bracket(bracket, team_stats_df, model, outfile):
    if len(bracket) == 1:
        outfile.write(f"National Champion {bracket[0]}")
        return
    else:
        outfile.write(f"Round of {len(bracket)}\n")
        outfile.write(f"------------------------------\n")
        winners = []
        for i in range(0, len(bracket), 2):
            team1 = bracket[i]
            team2 = bracket[i+1]

            result1 = compute_expected_result(team1, team2, team_stats_df, model)
            result2 = compute_expected_result(team2, team1, team_stats_df, model)
            result = (result1 + 1 - result2) / 2

            if result >= 0.5:
                winner = team1
                outfile.write("{} vs {} ==> {} ({:0.2f})\n".format(team1, team2, winner, result))
            else:
                winner = team2
                outfile.write("{} vs {} ==> {} ({:0.2f})\n".format(team1, team2, winner, 1-result))

            winners.append(winner)

        outfile.write("\n")
        compute_bracket(winners, team_stats_df, model, outfile)

def simulate_bracket(bracket, team_stats_df, model, outfile):

    if len(bracket) == 1:
        outfile.write(f"National Champion {bracket[0]}")
        return
    else:
        outfile.write(f"Round of {len(bracket)}\n")
        outfile.write(f"------------------------------\n")
        winners = []
        for i in range(0, len(bracket), 2):
            team1 = bracket[i]
            team2 = bracket[i+1]

            result1 = compute_expected_result(team1, team2, team_stats_df, model)
            result2 = compute_expected_result(team1, team2, team_stats_df, model)
            result = (result1 + result2) / 2

            if result >= random.random():
                winner = team1
                outfile.write("{} vs {} ==> {} ({:0.2f})\n".format(team1, team2, winner, result))
            else:
                winner = team2
                outfile.write("{} vs {} ==> {} ({:0.2f})\n".format(team1, team2, winner, 1-result))

            winners.append(winner)

        outfile.write("\n")
        simulate_bracket(winners, team_stats_df, model, outfile)

def individual_matchups(team_stats_df, model):
    while True:
        team1 = input("Please enter team1: ").strip()
        if not team1 or team1 == "quit":
            break
        team2 = input("Please enter team2: ").strip()
        
        print(f"Computing {team1} vs {team2}")

        team1_stats = team_stats_df.loc[team_stats_df["School"] == team1]
        team2_stats = team_stats_df.loc[team_stats_df["School"] == team2]

        print(team1_stats)
        print(team2_stats)

        result = compute_expected_result(team1, team2, team_stats_df, model)


        if result >= 0.5:
            print(f"{team1} should win with probability {result}")
        else:
            print(f"{team2} should win with probability {1-result}")
        print()

def main():
    arg = None

    if len(sys.argv) > 1:
        arg = sys.argv[1]

    team_stats_df = pd.read_csv("team_stats.csv")
    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    model = torch.load("best_model.pth")['model'].cuda()

    bracket = read_bracket("bracket.txt")

    if arg == "--sim":
        with open("simulated_bracket.txt", "w") as outfile:
            simulate_bracket(bracket, team_stats_df, model, outfile)
    elif arg == "--ind":
        individual_matchups(team_stats_df, model)

    with open("predicted_bracket.txt", "w") as outfile:
        compute_bracket(bracket, team_stats_df, model, outfile)

if __name__ == "__main__":
    main()
