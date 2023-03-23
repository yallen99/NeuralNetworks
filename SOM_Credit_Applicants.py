# %% # Comment used to make cells
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Train the SOM
# MiniSom is a copyright free online implementation made available for public use
from minisom import MiniSom
som = MiniSom(x = 20, y = 20, input_len = 15, sigma = 1.0, learning_rate = 0.5)     # sigma = diameter of the neighbourhood, learning_rate = how quick the map diverges/learns
som.random_weights_init(X)
som.train_random(X, num_iteration = 100)

# Plot the SOM on a grid
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colours = ['r', 'g']
for i, v in enumerate(X):
    w = som.winner(v)
    plot(w[0] + 0.5,
         w[1] + 0.5,
        markers[y[i]],
        markeredgecolor = colours[y[i]],
        markerfacecolor = 'None',
        markersize = 5,
        markeredgewidth = 2)
show()