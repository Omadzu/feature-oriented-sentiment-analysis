# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:08:26 2017

@author: Ophelie
From: https://www.kaggle.com/c/word2vec-nlp-tutorial

BOW + Logistic Regression (SVM)

Summary:
    - Collect, clean and parse data
    - Bag of words: create vector containing words and the number of times they
    appeared
    - Train a model based on Word2Vec algorithm. Labeled data not needed.
    
    
    
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
from itertools import chain

# Download text data sets, including stop words
from nltk.corpus import stopwords
import nltk.data

# Regular Expression
import re

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer

# Word2Vec
from gensim.models import word2vec

# ROC curve and confusion matrix and classification report
from sklearn.metrics import roc_curve, auc, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt

""" Global variables """
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
remove_stopwords = False

""" Methods """


def processTime(seconds, whatIsProcessed):
    # see : http://stackoverflow.com/questions/775049/python-time-seconds-to-hms

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    logging.info("Process time of %s : %d:%02d:%02d" %
                 (whatIsProcessed, h, m, s))


def review_to_wordlist(review):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words. Returns a list of words.

    # 1. Remove HTML
    review_text = bs(review).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # 5. Return a list of words
    return(words)


def review_to_sentences(review):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words

    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists)
    return sentences


def calculateParallel(functionToCall, listOfStrings, threads=mp.cpu_count()):
    # see : https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
    # see : http://stackoverflow.com/questions/32789086/using-pool-map-to-apply-function-to-list-of-strings-in-parallel
    # see : http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments

    # Initialize an empty list to hold the clean reviews
    clean_strings = []

    logging.info("Distributed calculation started")
    pool = mp.Pool(threads)
    clean_strings = pool.map_async(functionToCall, listOfStrings)
    pool.close()

    return clean_strings


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


if __name__ == '__main__':

    # Config of logging module
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    # Read data from files
    train = pd.read_csv("../../data/labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3, encoding="utf-8")
    test = pd.read_csv("../../data/testData.tsv", header=0, delimiter="\t",
                       quoting=3, encoding="utf-8")
    unlabeled_train = pd.read_csv("../../data/unlabeledTrainData.tsv",
                                  header=0,
                                  delimiter="\t", quoting=3, encoding="utf-8")

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, "\
          "and %d unlabeled reviews\n" % (train["review"].size,
                                          test["review"].size,
                                          unlabeled_train["review"].size)

    start_time = time.time()

    # ======================================================
    # Multiprocessing parsing
    # ======================================================

    # Spread the calculation for the train set
    logging.info("Parsing sentences from training set")
    mapResults = calculateParallel(review_to_sentences, train["review"])
    processProgress(mapResults)
    sentences = mapResults.get()

    # Spread the calculation for the unlabeled set
    logging.info("Parsing sentences from unlabeled set")
    mapResults = calculateParallel(review_to_sentences,
                                   unlabeled_train["review"])
    processProgress(mapResults)
    unlabeled_sentences = mapResults.get()

    # Group all sentences
    sentences += unlabeled_sentences

    # Get a list of sentences only
    sentences = list(chain.from_iterable(sentences))

    # Display some information about sentences built (list of words)
    logging.debug("Number of sentences : %d " % len(sentences))
    logging.debug(sentences[0])
    logging.debug(sentences[1])

    total_time = time.time() - start_time
    processTime(total_time, "Clean and parse the data")

    # ======================================================
    # Training Word2Vec
    # ======================================================

    start_time = time.time()

    # Set values for various parameters
    num_features = 300            # Word vector dimensionality
    min_word_count = 40           # Minimum word count
    num_workers = mp.cpu_count()  # Number of threads to run in parallel
    context = 10                  # Context window size
    downsampling = 1e-3           # Downsample setting for frequent words

    # Initialize and train the model (take some time)
    logging.info("Training the model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using
    # Word2Vec.load()
    model_name = "results/300features_40minwords_10context"
    model.save(model_name)

    total_time = time.time() - start_time
    processTime(total_time, "Training Word2Vec")

    # ======================================================
    # Exploring the model's results
    # ======================================================

    # Deduce which word in a set is most dissimilar from the others
    words = "man woman child kitchen"
    logging.debug("From the list '%s' which word is the most dissimilar ? %s" %
                  (words, model.doesnt_match(words.split())))

    words = "france england germany berlin"
    logging.debug("From the list '%s' which word is the most dissimilar ? %s" %
                  (words, model.doesnt_match(words.split())))

    words = "paris berlin london austria"
    logging.debug("From the list '%s' which word is the most dissimilar ? %s" %
                  (words, model.doesnt_match(words.split())))

    # Deduce words which are the most similar to another
    word = "man"
    logging.debug("Which word is the most similar to '%s'? %s" %
                  (word, model.most_similar(word)))

    word = "queen"
    logging.debug("Which word is the most similar to '%s'? %s" %
                  (word, model.most_similar(word)))

    word = "awful"
    logging.debug("Which word is the most similar to '%s'? %s" %
                  (word, model.most_similar(word)))
