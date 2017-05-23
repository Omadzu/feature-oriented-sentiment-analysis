# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:11:24 2017

@author: Ophelie
from : http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from : http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

RNN algorithm for Sentiment Analysis
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging

"""GLOBAL VARIABLES"""
pima_indians_diabetes_path = "../../data/pima-indians-diabetes.csv"

# Fix random seed for reproductibility
seed = 7
np.random.seed(seed)

if __name__ == '__main__':
    # Config of logging module
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    # Load pima indians dataset
    dataset = np.loadtxt(pima_indians_diabetes_path, delimiter=",")

    # Split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # Define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    for train, test in kfold.split(X, Y):
        # Create a model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        # Fit the model
        model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)

        # Evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        logging.debug("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    logging.debug("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores),
                                           np.std(cvscores)))
