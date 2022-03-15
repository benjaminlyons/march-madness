import torch
import torch.nn as nn

from progress.bar import Bar

import numpy as np
import random

import sys
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DROP = 0.5
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(30, 32),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

# first we will read and prepare the data
random.seed(7)
stats = []
results = []
with open("training_data.csv", "r") as f:
    rows = []
    for row in f.readlines():
        row = [float(x) for x in row.strip().split(',')]
        rows.append(row)

    random.shuffle(rows)
    for row in rows:
        stats.append(np.array(row[:-1]))
        results.append(np.array([row[-1]]))

def evaluate(model, inputs, results, loss_fn):
    model.eval()
    permutation = torch.randperm(inputs.size()[0])
    eval_loss = 0
    incorrect = 0
    for i in range(0, inputs.size()[0], batch_size):
        indices = permutation[i:min(i+batch_size, inputs.size()[0])]
        x, y = inputs[indices], results[indices]
        outputs = model.forward(x)
        loss = loss_fn(outputs, y)
        eval_loss += loss.item()

        diff = torch.round(outputs)
        diff = torch.subtract(diff, y)
        diff = torch.abs(diff)
        incorrect += torch.sum(diff).item()
    accuracy = (inputs.size()[0] - incorrect) / inputs.size()[0]
    # print(f"Accuracy: {accuracy}")
    return eval_loss / inputs.size()[0], accuracy

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

# now create the model and optimizers
batch_size = 256
EPOCHS = 500

load = False
eval = False
if len(sys.argv) > 1 and sys.argv[1] == "--load":
    load = True
elif len(sys.argv) > 1 and sys.argv[1] == "--eval":
    eval = True

if load or eval:
    saved_state = torch.load("model.pth")
    model = saved_state["model"].cuda()
    optimizer = saved_state["optimizer"]
    starting_epoch = saved_state["epoch"]
else:
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    starting_epoch = 0

loss_fn = nn.MSELoss()

if eval:
    test_loss = evaluate(model, testing_data, testing_outputs, loss_fn)
    print(f"Testing Loss: {test_loss}")
    sys.exit(0)


loss_log = open("loss.csv", "w")
for epoch in range(starting_epoch, EPOCHS):
    model.train()
    prog = Bar(f"Training Epoch {epoch}:", max=int(training_size / batch_size) + 1)
    permutation = torch.randperm(training_data.size()[0])
    train_loss = 0
    for i in range(0, training_data.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:min(i+batch_size, training_data.size()[0])]

        x, y = training_data[indices], training_outputs[indices]

        # print(x.shape)
        # print(y.shape)
        outputs = model.forward(x)
        loss = loss_fn(outputs, y)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        prog.next()
        

    prog.finish()
    train_loss = train_loss / training_size

    # now test on validation data
    model.eval()
    val_loss, val_accuracy = evaluate(model, validation_data, validation_outputs, loss_fn)
    training_loss, train_accuracy = evaluate(model, training_data, training_outputs, loss_fn)
    print(f"Training loss: {train_loss}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Validation loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    # time.sleep(1)

    loss_log.write(f"{epoch},{train_loss},{val_loss},{train_accuracy},{val_accuracy}\n")
    if epoch % 10 == 0:
        torch.save({"model": model, "optimizer": optimizer, "epoch": epoch}, "model.pth")

torch.save({"model": model, "optimizer": optimizer, "epoch": epoch}, "model.pth")
