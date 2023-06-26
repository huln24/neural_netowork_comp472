import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Read training data
dataset = pd.read_csv("train.csv", header=0, dtype=np.int32)
X = dataset.values[:, 0:5]
y = dataset.values[:, 5]

# Convert pandas array to PyTorch tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Construct our model
model = nn.Sequential(nn.Linear(5, 1000), nn.ReLU(), nn.Linear(1000, 1), nn.Sigmoid())

# Initialize loss function and Optimizer (gradient descent)
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.001)

n_epochs = 1000  # n. of iterations to go through dataset to train
batch_size = 6  # n. of data in each batch

# train the neural network
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i : i + batch_size]
        y_pred = model(Xbatch)

        ybatch = y[i : i + batch_size]
        loss = loss_fn(y_pred, ybatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Finished epoch {epoch}, latest loss {loss}")
print(f"Finished: latest loss {loss}\n")


# Read test dataset
test_set = pd.read_csv("test.csv", header=0, dtype=np.int32)
X_test = test_set.values[:, 0:5]
y_test = test_set.values[:, 5]


# Convert pandas array to PyTorch tensor
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# compute accuracy
with torch.no_grad():
    y_pred = model(X_test)

accuracy = (y_pred.round() == y_test).float().mean()
print(f"Accuracy {accuracy}\n")

# print prediction vs. expected values
for i in range(len(y_test)):
    print("%s => %d (expected %d)" % (X_test[i].tolist(), y_pred[i].round(), y_test[i]))
