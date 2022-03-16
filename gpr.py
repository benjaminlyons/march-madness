from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import random

import pickle

def main():
    # load the data
    random.seed(7)
    stats = []
    spreads = []
    with open("spread_training_data.csv", "r") as f:
        rows = []
        for row in f.readlines():
            row = [float(x) for x in row.strip().split(',')]
            rows.append(row)
        random.shuffle(rows)
        for row in rows:
            stats.append(np.array(row[:-1]))
            spreads.append(np.array(row[-1]))

    size = len(stats)
    training_size = int(size*.8)
    validation_size = size - training_size

    training_data = np.array(stats[:training_size])
    training_outputs = np.array(spreads[:training_size])
    validation_data = np.array(stats[training_size:])
    validation_outputs = np.array(spreads[training_size:])

    kernel = 2*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 100)) 
    noise = 4
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2,  n_restarts_optimizer=9)
    gp.fit(training_data, training_outputs)

    mean_prediction, std_prediction = gp.predict(validation_data, return_std=True)
    avg_error = np.sum(np.abs(np.subtract(mean_prediction, validation_outputs)))/len(validation_outputs)

    print(mean_prediction)
    print(validation_outputs)
    print(std_prediction)
    print(avg_error)

    with open("gp_model.pkl", "wb") as f:
        pickle.dump(gp, f)
        

if __name__ == "__main__":
    main()
