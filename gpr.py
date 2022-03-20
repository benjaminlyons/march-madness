from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import numpy as np
import random

import pickle
import sys
import joblib


def main():
    # load the data
    random.seed(7)
    stats = []
    spreads = []

    if len(sys.argv) == 1:
        print("Need arguments.")
        sys.exit(0)

    if sys.argv[1] == "--train":
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

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01,100000.0))*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10000))  + WhiteKernel(noise_level=1.0, noise_level_bounds=(0.1, 10000))
        gp = GaussianProcessRegressor(kernel=kernel,  n_restarts_optimizer=9)
        gp.fit(training_data, training_outputs)

        mean_prediction, std_prediction = gp.predict(validation_data, return_std=True)
        avg_error = np.sum(np.abs(np.subtract(mean_prediction, validation_outputs)))/len(validation_outputs)

        print(mean_prediction)
        print(validation_outputs)
        print(std_prediction)
        print(avg_error)

        with open("gp_model.pkl", "wb") as f:
            pickle.dump(gp, f)
            # joblib.dump(gp, f, compress=3)

    elif sys.argv[1] == "--eval":
        with open("gp_model.pkl", "rb") as f:
            gp = pickle.load(f)

        print(gp.kernel_)

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

        mean_prediction, std_prediction = gp.predict(validation_data, return_std=True)
        mae = np.sum(np.abs(np.subtract(mean_prediction, validation_outputs)))/len(validation_outputs)
        rmse = np.sqrt(np.sum(np.square(np.subtract(mean_prediction, validation_outputs))) / len(validation_outputs))
        mean = np.sum(validation_outputs) / len(validation_outputs)
        stdev = np.sqrt(np.sum(np.square(np.subtract(mean, validation_outputs))) / len(validation_outputs))
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"General: {mean} +- {stdev}")
        print(f"Score: {gp.score(training_data, training_outputs)}")

        spreads = mean_prediction.tolist()

        correct = 0
        for s, r in zip(spreads, validation_outputs):
            if s > 0 and r > 0 or s < 0 and r < 0:
                correct += 1
        acc = correct / len(spreads)

        print(f"Classification Accuracy: {acc}")


if __name__ == "__main__":
    main()
