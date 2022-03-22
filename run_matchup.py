import pandas as pd
import torch
import numpy as np
import random
import sys
import pickle
from nn import Net
from ensemble import GamePredictor

STATS = ["W-L%", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

# reads bracket from filename, then computes results
def read_bracket(filename):
    bracket = []
    with open(filename, 'r') as f:
        for team in f.readlines():
            bracket.append(team.strip())
    return bracket

def compute_bracket(bracket, team_stats_df, predictor, outfile):
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
            
            result, gp_pred, gp_std = predictor.predict(team1, team2, home=False)

            if result >= 0.5:
                winner = team1
                outfile.write("{} vs {} ==> {} ({:0.2f}) [{:0.2f}]\n".format(team1, team2, winner, result, -gp_pred))
            else:
                winner = team2
                outfile.write("{} vs {} ==> {} ({:0.2f}) [{:0.2f}]\n".format(team1, team2, winner, 1-result, gp_pred))

            winners.append(winner)

        outfile.write("\n")
        compute_bracket(winners, team_stats_df, predictor, outfile)

def simulate_bracket(bracket, team_stats_df, predictor, outfile):

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

            predictor.predict(team1, team2, home=False)

            win_team1 = 0
            win_team2 = 0
            for i in range(7):
                if result >= random.random():
                    win_team1 += 1
                else:
                    win_team2 += 1

            if win_team1 > win_team2:
                winner = team1
                outfile.write("{} vs {} ==> {} ({:0.2f}) [{:0.2f}]\n".format(team1, team2, winner, result, -gp_pred))
            else:
                winner = team2
                outfile.write("{} vs {} ==> {} ({:0.2f}) [{:0.2f}]\n".format(team1, team2, winner, 1-result, gp_pred))

            winners.append(winner)

        outfile.write("\n")
        simulate_bracket(winners, team_stats_df, predictor, outfile)

def individual_matchups(team_stats_df, predictor):
    while True:
        team1 = input("Please enter team1: ").strip()
        if not team1 or team1 == "quit":
            break
        team2 = input("Please enter team2: ").strip()
        
        print(f"Computing {team1} vs {team2}")

        result, gp_pred, gp_std = predictor.predict(team1, team2, home=True)

        if result >= 0.5:
            win_odds = -100*result / (1-result)
            lose_odds = 100*result / (1-result)
            print("{} should win with probability {:0.3f}".format(team1, result))
            print("Favored by {:0.2f} +/- {:0.2f} points".format(gp_pred, gp_std))
            print("{} {}".format(team1, int(win_odds)))
            print("{} +{}".format(team2, int(lose_odds)))
        else:
            win_odds = -100*(1-result) / result
            lose_odds = 100*(1-result) / result
            print("{} should win with probability {:0.3f}".format(team2, 1-result))
            print("Favored by {:0.2f} +/- {:0.2f} points".format(-gp_pred, gp_std))
            print("{} {}".format(team2, int(win_odds)))
            print("{} +{}".format(team1, int(lose_odds)))

        print()

def main():
    arg = None
    filename = "bracket.txt"
    if len(sys.argv) > 1:
        arg = sys.argv[1]

    team_stats_df = pd.read_csv("team_stats.csv")
    team_stats_df[STATS] = (team_stats_df[STATS] - team_stats_df[STATS].mean()) / team_stats_df[STATS].std()
    nn_model = torch.load("best_model.pth")['model'].cuda()
    nn_model.eval()

    bracket = read_bracket(filename)

    predictor = GamePredictor("best_model.pth", "gp_model.pkl")

    with open("gp_model.pkl", "rb") as f:
        gp_model = pickle.load(f)

    if arg == "--sim":
        with open("simulated_bracket.txt", "w") as outfile:
            simulate_bracket(bracket, team_stats_df, predictor,  outfile)
        sys.exit(0)
    elif arg == "--ind":
        individual_matchups(team_stats_df, predictor)
        sys.exit(0)

    if len(sys.argv) > 1:
        filename = arg
    bracket = read_bracket(filename)

    with open(f"predicted_{filename}", "w") as outfile:
        compute_bracket(bracket, team_stats_df, predictor, outfile)

if __name__ == "__main__":
    main()
