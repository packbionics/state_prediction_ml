'''
This is a script that will perform the actions required to get megatox aim 2 finished.
First, we will load in the datasets (training and test) for toxicity models. Then we will load in
The superdrug library. Our goals are thus;
1: R-group decomposition. This will be followed by tanimoto of fragments and core between....superdrug an tox?
2: murcko? Maybe.
3: similarity maps from predictions I think? Will need to flesh this out a bit more.
'''

# load in the libraries
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def plot_tsne(df, feature_col, label_col):
    seed = 42
    
    X = np.array([])
    labels = np.array([])
    
    for feature, label in zip(df[feature_col], df[label_col]):
        X = np.append(X, feature)
        labels = np.append(labels, label)
    
    X = X.reshape(-1, 1)
    
    print(X.shape)
    
    # Save the full t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    X_fit = tsne.fit_transform(X)
    df['tsne_x'] = X_fit.T[0]
    df['tsne_y'] = X_fit.T[1]
    print(df['tsne_x'])
    
    fig = plt.figure()
    plt.scatter(df['tsne_x'], df['tsne_y'])
    plt.show()
    # plt.scatter(fin_suppl['tsne_x'], fin_suppl['tsne_y'])
    # return fig
    # fin_suppl.to_csv('tsne_test.csv', index=False)