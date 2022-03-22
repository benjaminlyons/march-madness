import pandas as pd
import numpy as np

# we are assuming the following model
# Obs = Adv + Noise
# => P(Adv | Obs) oc P(Obs | Adv)*P(Adv)
# Assume P(Adv) = N(prior_mean, prior_var)
# we claim that each homefield adv is gaussian
def posterior(prior_mean, prior_var, sigma_n2, obs):
    posterior_mean = (obs*prior_var + prior_mean*sigma_n2)/ (prior_var + sigma_n2)
    posterior_var = (prior_var*sigma_n2)/(prior_var + sigma_n2)
    return posterior_mean, posterior_var

# derived via maximum likelihood estimation
# returns the variance due to noise
def estimate_noise(prior_mean, prior_var, data):
    first_term = np.sum(np.square(np.subtract(data, prior_mean))) / len(data)
    sigma_n2 = first_term - prior_var
    return sigma_n2

def compute_homefield_advs():
    prior_mean = 3.5
    prior_var = 1.5
    games_df = pd.read_csv("data/all_game_logs.csv")
    teams_df = pd.read_csv("data/all_team_stats.csv")

    teams = set(teams_df["School"])

    homefield_diffs = {}

    for team in teams:
        games = games_df.loc[games_df["school1"] == team]
        home_games = games.loc[games["location"] == 1]
        neutral_games = games.loc[games["location"] == 0]
        away_games = games.loc[games["location"] == -1]

        home_dif = sum(home_games["score1"] - home_games["score2"])
        neutral_dif = sum(neutral_games["score1"] - neutral_games["score2"])
        away_dif = sum(away_games["score1"] - away_games["score2"])

        home_avg = home_dif / len(home_games)
        away_avg = away_dif / len(away_games)

        if len(neutral_games):
            neutral_avg = neutral_dif / len(neutral_games)

        adv = (home_avg - away_avg) / 2.0

        homefield_diffs[team] = adv

    data = np.array(list(homefield_diffs.values()))
    sigma_n2 = estimate_noise(prior_mean, prior_var, data)

    homefield_adv = {}
    for team, val in homefield_diffs.items():
        adv, std = posterior(prior_mean, prior_var, sigma_n2, val)
        homefield_adv[team] = adv
    advs = np.array(list(homefield_adv.values()))
    new_mean = np.mean(advs)
    new_var = np.var(advs)

    return homefield_adv

if __name__ == "__main__":
    main()
