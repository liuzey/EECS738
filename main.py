# This is for Project1 - Probably Interesting Data in course EECS738 Machine Learning.
# Written by Zeyan Liu (StudentID: 3001190).
# Run command example: {python main.py iris KMEANS -k 3} or {python main.py zoo GMM -o True}

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from algorithms import gmm, kde, kmeans, knn


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help='Dataset for analysis.')
parser.add_argument("algorithm", type=str, help='Algorithm: {KDE, GMM, KNN, KMEANS}.')
parser.add_argument("-k", type=int, default=3, help='K for KNN or K-means.')
parser.add_argument("-r", type=float, default=0.2, help='Validation split ratio.')
parser.add_argument("-o", "--optimized", type=bool, default=False, help='Optimization for cluster number in KMEANS and GMM')
args = parser.parse_args()

DATA_DIR = './data/' + args.data
models = {'KDE': kde.KDE, 'GMM': gmm.GMM, 'KNN': knn.KNN, 'KMEANS': kmeans.KMEANS}
if not os.path.exists(DATA_DIR):
    raise Exception('Invalid Directory.')


def load_data():
    filename = glob.glob(DATA_DIR+'/*.csv')[0]
    # Multiple data files are not merged in this project.
    data_item = pd.read_csv(filename)
    print(filename, data_item.shape)
    print(data_item.head(3))
    # print(data_item.dtypes)
    print(data_item.columns)
    # print(data_item.describe())
    # one-hot.
    mapping = {}
    labels = data_item.iloc[:, -1].value_counts().index
    for i in range(len(labels)):
        mapping[labels[i]] = i
    print(mapping)
    data_item.iloc[:, -1] = data_item.iloc[:, -1].map(mapping)

    for redundant in ['Id', 'animal_name']:
        try:
            del data_item[redundant]
        except:
            pass

    return data_item


def statistics_overview(dataset):
    # Class Distribution
    print(dataset.groupby(dataset.columns[-1]).size())
    # Draw Box Plots
    dataset.iloc[:,:-1].plot(kind='box', subplots=True, layout=(1, len(dataset.columns)-1), sharex=False, sharey=False)
    # Draw Histograms
    dataset.iloc[:,:-1].hist()
    # Draw Scatter Plots
    pd.plotting.scatter_matrix(dataset.iloc[:,:-1])
    plt.show()


if __name__ == '__main__':
    data = load_data()
    print('\n-----Statistics & Plots-----')
    # statistics_overview(data)

    print('\n-----Algorithm Begins-----')
    start = time.time()
    # model = models[args.algorithm](data, args.r, k=args.k, norm=True, sklearn_valid=True, optimized=False)
    model = models[args.algorithm](data, args.r, k=args.k, norm=True, sklearn_valid=True, optimized=args.optimized)
    model.run()
    print('\n-----Time Used: {}s-----'.format(time.time()-start))