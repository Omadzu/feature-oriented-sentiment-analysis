# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:08:26 2017

@author: Ophelie
From: https://www.kaggle.com/c/word2vec-nlp-tutorial


Summary:
    - Collect, clean and parse data
    - Train a model based on Word2Vec algorithm. Labeled data not needed.
    - Word2Vec has understood quite well the semantic of the sentences
    used inside the review and is capable of deducing similarity and
    dissimilarity between words with a certain accuracy
    - Use a Random Forest to predict values based on previous Word2Vec
"""

import os.path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import multiprocessing as mp
import time
import logging
from itertools import chain

# Download text data sets, including stop words
from nltk.corpus import stopwords
import nltk.data

# Regular Expression
import re

# Word2Vec
from gensim.models import word2vec, Word2Vec

# Random forest
from sklearn.ensemble import RandomForestClassifier

# ROC curve and confusion matrix and classification report
from sklearn.metrics import roc_curve, auc, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt

""" Global variables """
# Path to set
train_set_path = "../../data/labeledTrainData.tsv"
test_set_path = "../../data/testData.tsv"
unlabeled_train_set_path = "../../data/unlabeledTrainData.tsv"
word2vec_model_path = "../../results/binary-classification-approches/300features_40minwords_10context"
word2vec_prediction_results_path = "../../results/binary-classification-approches/Word2Vec_AverageVectors.csv"

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
remove_stopwords = True


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


def calculateParallel(functionToCall, iterable, threads=mp.cpu_count()):
    # see : https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
    # see : http://stackoverflow.com/questions/32789086/using-pool-map-to-apply-function-to-list-of-strings-in-parallel
    # see : http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments

    # Initialize an empty list to hold the clean reviews
    listOfResults = []

    logging.info("Distributed calculation started")
    pool = mp.Pool(threads)
    listOfResults = pool.map_async(functionToCall, iterable)
    pool.close()

    return listOfResults


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


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    # words : words composing the review/paragraph

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0.

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    # Loop through the reviews
    for review in reviews:

        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            logging.info("Review %d of %d" % (counter, len(reviews)))

        # Call the function that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,
                                                    num_features)

        # Increment the counter
        counter = counter + 1.

    return reviewFeatureVecs


if __name__ == '__main__':

    # Config of logging module
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    
    # Read data from files
    train = pd.read_csv(train_set_path, header=0,
                        delimiter="\t", quoting=3, encoding="utf-8")
    test = pd.read_csv(test_set_path, header=0, delimiter="\t",
                       quoting=3, encoding="utf-8")
    unlabeled_train = pd.read_csv(unlabeled_train_set_path,
                                  header=0,
                                  delimiter="\t", quoting=3,
                                  encoding="utf-8")

    # If the model exists already, skip the steps of cleaning, parsing the data
    # and training Word2Vec, load the existing model instead
    if not os.path.isfile(word2vec_model_path):

        # Verify the number of reviews that were read (100,000 in total)
        logging.debug("Read %d labeled train reviews, %d labeled test "
                      "reviews, and %d unlabeled reviews\n" %
                      (train["review"].size,
                       test["review"].size,
                       unlabeled_train["review"].size))

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
        model_name = word2vec_model_path
        model.save(model_name)

        total_time = time.time() - start_time
        processTime(total_time, "Training Word2Vec")

    else:
        model = Word2Vec.load(word2vec_model_path)

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

    # The Word2Vec model trained consists of a feature vector for each word
    # in the vocabulary, stored in a numpy array "syn0"
    logging.debug("What does the model look like ? %s" % type(model.wv.syn0))
    logging.debug("Shape : %s" % (model.wv.syn0.shape,))

    # The number of rows in syn0 is the number of words in the vocabulary
    # The number of columns is the size of the feature vector
    # Setting the minimum word count to 40 gave a total vocaulary of 16,490
    # words
    # Here is an example of a word vector
    logging.debug("Vector for the word 'flower' : %s" % (model["flower"],))

    # ======================================================
    # Vector Averaging
    # ======================================================

    # Reviews do not have the same length. Have to find a way to take
    # individual word vectors and transform them into a feature set which has
    # the same length for every review
    # Each word is a vector in 300-dimensional space, we can use vector
    # operations to combine the words in each review. Here we try vector
    # averaging, we average the word vectors in a given review

    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.

    # Averaging vectors for training set
    logging.info("Creating average feature vectors for the training reviews\
                 set")

    start_time = time.time()

    # Spread the calculation for the train set
    logging.info("Parsing sentences from training set")
    mapResults = calculateParallel(review_to_wordlist, train["review"])
    processProgress(mapResults)
    clean_train_reviews = mapResults.get()

    total_time = time.time() - start_time
    processTime(total_time, "Clean training reviews")

    start_time = time.time()

    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    total_time = time.time() - start_time
    processTime(total_time, "Calculate average feature vector for training\
                set")

    # Averaging vectors for training set
    logging.info("Creating average feature vectors for the testing reviews\
                 set")

    start_time = time.time()

    # Spread the calculation for the test set
    logging.info("Parsing sentences from testing set")
    mapResults = calculateParallel(review_to_wordlist, test["review"])
    processProgress(mapResults)
    clean_test_reviews = mapResults.get()

    total_time = time.time() - start_time
    processTime(total_time, "Clean testing reviews")

    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    total_time = time.time() - start_time
    processTime(total_time, "Calculate average feature vector for test set")

    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    logging.info("Fitting a random forest to labeled training data...")

    start_time = time.time()

    forest = forest.fit(trainDataVecs, train["sentiment"])

    total_time = time.time() - start_time
    processTime(total_time, "Fitting the forest")

    # Test & extract results
    result = forest.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(word2vec_prediction_results_path, index=False, quoting=3)
