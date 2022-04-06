from nn import Net
from scipy.stats import norm
import torch
import pickle
import random
import numpy as np
from compute_homefield_adv import compute_homefield_advs
import pandas as pd

import math
STATS = ["W-L%", "SOS", "Pace", "ORtg", "FTr", "3PAr", "TS%", "TRB%", "AST%", "STL%", "BLK%", "eFG%", "TOV%", "ORB%"]

class GamePredictor():
    def __init__(self, nn_path, gp_path):
        self.nn_model = torch.load("best_model.pth")['model'].cuda()
        self.nn_model.eval()

        with open("gp_model.pkl", 'rb') as f:
            self.gp_model = pickle.load(f)

        self.homefield_adv, self.home_adv_probs = compute_homefield_advs()

        self.team_stats = pd.read_csv("team_stats.csv")
        self.team_stats[STATS] = (self.team_stats[STATS] - self.team_stats[STATS].mean()) / self.team_stats[STATS].std()

    # if home is true, then team1 is the home team
    # if home is false, then the game is played at a neutral site
    def predict(self, team1, team2, home=False, year=2022):
        team1_stats = self.team_stats.loc[self.team_stats["School"] == team1].loc[self.team_stats["Year"] == year]
        team2_stats = self.team_stats.loc[self.team_stats["School"] == team2].loc[self.team_stats["Year"] == year]

        team1_stats = team1_stats[STATS].to_numpy()
        team2_stats = team2_stats[STATS].to_numpy()

        X = np.concatenate((team1_stats, team2_stats), axis=1)

        home_adv = 0
        home_var = 0
        if home:
            home_adv, home_var = self.homefield_adv[team1]

        # put both teams in both orders and take the avg
        nn_prob1, gp_prob1, gp_pred1, gp_std1 = self.__predict(X, home_adv, home_var)
        X = np.concatenate((team2_stats, team1_stats), axis=1)
        nn_prob2, gp_prob2, gp_pred2, gp_std2 = self.__predict(X, -home_adv, home_var)

        nn_prob = (nn_prob1 + 1 - nn_prob2) / 2
        gp_prob = (gp_prob1 + 1 - gp_prob2) / 2
        gp_pred = (gp_pred1 - gp_pred2) / 2
        gp_std  = (gp_std1 + gp_std2) /2 
        result = (nn_prob + gp_prob) / 2

        return result, gp_pred, gp_std

    # consider adding home field advantage variance as well
    def __predict(self, X, home_adv=0, home_var=0):
        gp_pred, gp_std = self.gp_model.predict(X, return_std=True)
        gp_pred += home_adv
        gp_std = math.sqrt(gp_std**2 + home_var)

        # compute probability of gp prediction
        # first find the z_score of 0
        # take the negative value of it
        z_score = gp_pred[0] / gp_std
        gp_prob = norm.cdf(z_score)

        # assumes neutral site
        # if not home_adv and not home_var:
        #     X = np.concatenate((np.array([[0]]), X), axis=1)
        # elif home_adv > 0:
        #     X = np.concatenate((np.array([[1]]), X), axis=1)
        # else:
            # X = np.concatenate((np.array([[-1]]), X), axis=1)
        X = torch.tensor(X, dtype=torch.float).cuda()
        nn_y = self.nn_model(X).item()

        # now, adjust nn_y based on bayes theorem
        prob_win_home = self.home_adv_probs[0]
        prob_loss_home = self.home_adv_probs[1]

        adj_nn_prob = nn_y
        # if it is a home game
        if home_adv > 0:
            adj_nn_prob = (prob_win_home * nn_y) / (prob_win_home*nn_y + prob_loss_home*(1-nn_y))
        # if it is an away game
        elif home_adv < 0:
            adj_nn_prob = (prob_loss_home * nn_y) / (prob_loss_home*nn_y + prob_win_home*(1-nn_y))

        # print("NN:", adj_nn_prob)
        # print("GP:", gp_prob)
        return adj_nn_prob, gp_prob, gp_pred[0], gp_std

    def test_classification(self, X, y):
        count = 0
        correct = 0
        for inp, res in zip(X, y):
            loc = inp[0]
            inp = inp[1:]
            if loc > 0:
                pred, _, _ = self.__predict(inp, )        


def main():
    predictor = GamePredictor("best_model.pth", "gp_model.pkl")
    team1 = "Alabama"
    team2 = "Notre Dame"

    res, pred, std = predictor.predict(team1, team2, False)

    print("{} win prob: {:0.2f}".format(team1, res))
    print("Favored by {:0.2f} points".format(pred))

if __name__ == "__main__":
    main()
