from nn import Net
from scipy.stats import norm
import torch
import pickle
import random
import numpy as np

def read_data(filename):
    # load data
    random.seed(722)
    stats = []
    results = []
    with open(filename, "r") as f:
        rows = []
        for row in f.readlines():
            row = [float(x) for x in row.strip().split(',')]
            rows.append(row)

        random.shuffle(rows)
        for row in rows:
            stats.append(np.array([row[:-1]]))
            results.append(np.array([row[-1]]))

    stats = torch.tensor(np.array(stats), dtype=torch.float).cuda()
    results = torch.tensor(np.array(results), dtype=torch.float).cuda()

    # now divide into different types
    size = stats.shape[0]
    training_size = int(size*.6)
    validation_size = int(size*.2)
    testing_size = size - training_size - validation_size

    training_data = stats[:training_size]
    training_outputs = results[:training_size]
    validation_data = stats[training_size:training_size+validation_size]
    validation_outputs = results[training_size:training_size+validation_size]
    testing_data = stats[training_size+validation_size:]
    testing_outputs = results[training_size+validation_size:]

    return (training_data, training_outputs, validation_data, validation_outputs, testing_data, testing_outputs)

def ensemble_predict(X, nn_model, gp_model):
    gp_pred, gp_std = gp_model.predict(X, return_std=True)

    # compute probability of gp prediction
    # first find the z_score of 0
    # take the negative value of it
    z_score = gp_pred[0] / gp_std[0]
    gp_prob = norm.cdf(z_score)


    X = torch.tensor(X, dtype=torch.float).cuda()
    nn_y = nn_model(X).item()


    return nn_y, gp_prob, gp_pred[0], gp_std[0]

def main():

    # load the models
    nn_model = torch.load("best_model.pth")['model'].cuda()
    nn_model.eval()

    with open("gp_model.pkl", 'rb') as f:
        gp_model = pickle.load(f)

    _, _, _, _, testing_data, testing_output = read_data("training_data.csv")

    testing_data = testing_data.cpu().detach().numpy()
    testing_output = testing_output.cpu().detach().numpy().tolist()

    correct = 0
    for X, y in zip(testing_data, testing_output):
        nn_y, gp_prob, gp_pred, gp_std = ensemble_predict(X, nn_model, gp_model)
        result = 0.5*nn_y + 0.5*gp_prob
        if result > 0.5 and y[0] == 1 or result < 0.5 and y[0] == 0:
            correct += 1
    print(f"Accuracy: {correct/len(testing_output)}")

if __name__ == "__main__":
    main()
