# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:29:50 2017

@author: Ophelie
From: https://www.kaggle.com/c/word2vec-nlp-tutorial

BOW + Random Forest

Summary:
    - Collect, clean and parse data
    - Bag of words: create vector containing words and the number of times they
    appeared
    - Random Forest: The algorithm learns from the data for classification
    - Predictions are made  from another set of data with the Random Forest
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import multiprocessing as mp
import time
from sklearn.model_selection import train_test_split
import logging

# Download text data sets, including stop words
from nltk.corpus import stopwords

# Regular Expression
import re

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer

# Random forest
from sklearn.ensemble import RandomForestClassifier

# ROC curve and confusion matrix and classification report
from sklearn.metrics import roc_curve, auc, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt


""" Methods """


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)

    # 1. Remove HTML
    review_text = bs(raw_review).get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Join the words back into one string separated by space,
    #    and return the result.
    return (" ".join(meaningful_words))


def calculateParallel(listOfReviews, threads=mp.cpu_count()):
    # see : https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
    # see : http://stackoverflow.com/questions/32789086/using-pool-map-to-apply-function-to-list-of-strings-in-parallel

    # Initialize an empty list to hold the clean reviews
    clean_reviews = []

    logging.info("Distributed calculation started")
    pool = mp.Pool(threads)
    clean_reviews = pool.map_async(review_to_words, listOfReviews)
    pool.close()

    return clean_reviews


def processProgress(expectedResults):
    """
    expectedResults gathers all results of a distributed process in an
    asynchronous way. This function prints the progress of such a process.

    AsyncResult expectedResults
    """
    # see : http://stackoverflow.com/questions/26238691/counting-total-number-of-tasks-executed-in-a-multiprocessing-pool-during-executi

    numProcess = expectedResults._number_left
    while not expectedResults.ready():
        processesDone = numProcess - expectedResults._number_left
        pourcentage = 100.0 * (float(processesDone)/numProcess)
        print("{} %" .format(pourcentage))
        time.sleep(1)
    logging.info("Distributed calculation finished !")


def cleanAndParseReviews(listOfReviews):
    logging.info("Start of cleaning and parsing reviews")
    
    # Distribute on the processor cores the calculation
    mapResults = calculateParallel(listOfReviews)
    processProgress(mapResults)
    clean_train_reviews = mapResults.get()

    logging.info("End of cleaning and parsing reviews")
    return clean_train_reviews

def displayModelAccuracy(expected_results, predicted_results):
    # Display the confusion matrix
    matrix = confusion_matrix(expected_results, predicted_results)
    logging.info("Confusion matrix : %s" % (matrix,))
    
    # Display the classification report
    report = classification_report(expected_results, predicted_results)
    logging.info("Classification report : %s" % (report,))

    # Display the ROC curve
    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate, 'darkorange',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.show()

if __name__ == '__main__':

    # Config of logging module
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    # Collect and divide the data into training and testing sets
    data = pd.read_csv("../../data/labeledTrainData.tsv", header=0,
                       delimiter="\t", quoting=3)
    train_review, test_review, train_sentiment, test_sentiment = \
        train_test_split(data["review"], data["sentiment"], test_size=0.2)

    # Get the number of the reviews based on the dataframe column size
    train_set_num_reviews = train_review.size
    logging.debug("Number of reviews inside the training set : %d"
                  % train_set_num_reviews)

    logging.info("Cleaning and parsing the training set movie reviews...")

    # Clean and parse the training set of reviews
    clean_train_reviews = cleanAndParseReviews(train_review)

    logging.info("Creating bag of words...")

    # Initialize the CountVectorizer object, which is scikit-learn's
    # bag of words tool
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transforms() does 2 functions: first, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit-transform should be a list of
    # strings
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    if (logging.getLogger().getEffectiveLevel() == logging.DEBUG):
        logging.debug("Print the 10 first vocabulary words")
        index = 0
        while index < 10:
            logging.debug("The word '%s' of the vocabulary appeared %d times"
                          % (vocab[index], dist[index]))
            index += 1

    logging.info("Training the random forest...")

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_data_features, train_sentiment)

    logging.info("Cleaning and parsing the test set movie reviews...")

    # Clean and parse the testing set of reviews
    clean_test_reviews = cleanAndParseReviews(test_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentimental label predictions
    # see: https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
    predictions = forest.predict(test_data_features)
    actual = tuple(test_sentiment)

    # see: http://www.ultravioletanalytics.com/2014/12/16/kaggle-titanic-competition-part-x-roc-curves-and-auc/
    false_positive_rate, true_positive_rate, thresholds = \
        roc_curve(actual, forest.predict_proba(test_data_features)[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate)

    logging.debug("False positive rate : %s" % (false_positive_rate,))
    logging.debug("True positive rate : %s" % (true_positive_rate,))
    logging.debug("Thresholds : %s" % (thresholds,))
    logging.debug("ROC_AUC : %f" % roc_auc)

    logging.info("Prediction complete on test set !")

    displayModelAccuracy(actual, predictions)
