# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:35:49 2017

@author: Ophelie
From: https://www.kaggle.com/c/word2vec-nlp-tutorial

Word2Vec + Clustering + Random Forest Classifier

Summary:
    - Collect, clean and parse data
    - Train a model based on Word2Vec algorithm. Labeled data not needed.
    - Word2Vec has understood quite well the semantic of the sentences
    used inside the review and is capable of deducing similarity and
    dissimilarity between words with a certain accuracy
    - Use a Random Forest to predict values based on previous Word2Vec
    - Display a ROC Curve for Cross Validation using KFol (10 folds)
"""

import os.path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import multiprocessing as mp
import time
import logging
from itertools import chain
from scipy import interp

# Download text data sets, including stop words
from nltk.corpus import stopwords
import nltk.data

# Regular Expression
import re

# Word2Vec
from gensim.models import word2vec, Word2Vec

# Kmeans
from sklearn.cluster import KMeans

# Random forest
from sklearn.ensemble import RandomForestClassifier

# ROC curve and confusion matrix and classification report
from sklearn.metrics import roc_curve, auc, confusion_matrix,\
    classification_report
import matplotlib.pyplot as plt

# Method of Cross Validation : K-Fold
from sklearn.model_selection import StratifiedKFold

""" Global variables """
# Path to set
train_set_path = "../../data/labeledTrainData.tsv"
test_set_path = "../../data/testData.tsv"
unlabeled_train_set_path = "../../data/unlabeledTrainData.tsv"
word2vec_model_path = "../../results/binary-classification-approches/300features_40minwords_10context"

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
remove_stopwords = True

# Some parameters for Word2Vec
num_features = 300            # Word vector dimensionality
min_word_count = 40           # Minimum word count
num_workers = mp.cpu_count()  # Number of threads to run in parallel
context = 10                  # Context window size
downsampling = 1e-3           # Downsample setting for frequent words

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


def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    # Return the "bag of centroids"
    return bag_of_centroids


if __name__ == '__main__':

    whole_time = time.time()

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
    # Vector Quantization
    # ======================================================

    # Word2Vec creates clusters of semantically related words, so we can
    # exploit the similarity of words within a cluster. We find centers
    # of the word clusters with K-Means

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] / 5

#        logging.debug(word_vectors)
    logging.debug(num_clusters)

    # Initialize a k-means object and use it to extract centroids
    stepTitle = "Extract centroids"
    logging.info(stepTitle)

    start_time = time.time()
    kmeans_clustering = KMeans(n_clusters=num_clusters)

    # Sometimes Index error
    idx = kmeans_clustering.fit_predict(word_vectors)

    total_time = time.time() - start_time
    processTime(total_time, stepTitle)

    # Create a word / index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))

#    # For the first 10 clusters
#    for cluster in xrange(0, 10):
#        # Print the cluster number
#        logging.debug("Cluster %d" % cluster)
#
#        # Find all the words for that cluster number, and print them out
#        words = []
#        for i in xrange(0, len(word_centroid_map.values())):
#            if(word_centroid_map.values()[i] == cluster):
#                words.append(word_centroid_map.keys()[i])
#        logging.debug(words)

    forest = RandomForestClassifier(n_estimators=100)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    lw = 2

    # Divide the labeled set into training and testing set using stratified
    # K-fold
    # see : http://scikit-learn.org/stable/modules/cross_validation.html
    skfold = StratifiedKFold(n_splits=10)

    # Cross Validation based on KFold
    # see : http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    index = 0
    for train_index, test_index in skfold.split(train["review"],
                                                train["sentiment"]):

        logging.debug("[CROSS VALIDATION] Fold n: %d" % index)

        # Define the color of the curve for this fold
        color = colors[index]

        # Divide in training and testing set
        train_reviews_set, test_reviews_set =\
            train["review"][train_index], train["review"][test_index]
        train_polarity_set, test_polarity_set =\
            train["sentiment"][train_index], train["sentiment"][test_index]

        # Spread the calculation for the training set
        stepTitle = "Parsing and cleaning reviews from training set"
        logging.info(stepTitle)

        start_time = time.time()
        mapResults = calculateParallel(review_to_wordlist, train_reviews_set)
        processProgress(mapResults)
        clean_train_reviews = mapResults.get()

        total_time = time.time() - start_time
        processTime(total_time, stepTitle)

        # Spread the calculation for the testing set
        stepTitle = "Parsing and cleaning reviews from testing set"
        logging.info(stepTitle)

        start_time = time.time()
        mapResults = calculateParallel(review_to_wordlist, test_reviews_set)
        processProgress(mapResults)
        clean_test_reviews = mapResults.get()

        total_time = time.time() - start_time
        processTime(total_time, stepTitle)

        # Pre-allocate an array for the training set bags of centroids (for
        # speed)
        train_centroids = np.zeros((train_reviews_set.size, num_clusters),
                                   dtype="float32")

        # Transform the training set reviews into bags of centroids
        counter = 0
        for review in clean_train_reviews:
            train_centroids[counter] = create_bag_of_centroids(
                    review,
                    word_centroid_map)
            counter += 1

        # Repeat for test reviews
        test_centroids = np.zeros((test_reviews_set.size, num_clusters),
                                  dtype="float32")

        counter = 0
        for review in clean_test_reviews:
            test_centroids[counter] = create_bag_of_centroids(
                    review,
                    word_centroid_map)
            counter += 1

        # Fit random forest to the training data, using 100 trees
        stepTitle = "Fitting a random forest to labeled training data..."
        logging.info(stepTitle)

        start_time = time.time()
        forest = forest.fit(train_centroids, train_polarity_set)
        total_time = time.time() - start_time
        processTime(total_time, stepTitle)

        # Prediction step on the vectors of the testing set
        predictions = forest.predict(test_centroids)
        actual = tuple(test_polarity_set)

        false_positive_rate, true_positive_rate, thresholds = \
            roc_curve(actual, forest.predict_proba(test_centroids)[:, 1])

        mean_tpr += interp(mean_fpr, false_positive_rate, true_positive_rate)
        mean_tpr[0] = 0.0

        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, lw=lw, color=color,
                 label="ROC fold %d (area = %0.2f)" % (index, roc_auc))

        logging.debug("False positive rate :\n %s" % (false_positive_rate,))
        logging.debug("True positive rate :\n %s" % (true_positive_rate,))
        logging.debug("Thresholds :\n %s" % (thresholds,))
        logging.debug("ROC_AUC : %f" % roc_auc)

        # Display the confusion matrix
        matrix = confusion_matrix(actual, predictions)
        logging.info("Confusion matrix :\n %s" % (matrix,))

        # Display the classification report
        report = classification_report(actual, predictions)
        logging.info("Classification report :\n %s" % (report,))

        # raw_input("Press Enter to continue...")

        # Counter
        index += 1

    logging.info("Predictions complete on the %d test sets !" % (index))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')

    mean_tpr /= skfold.get_n_splits(train["review"], train["sentiment"])
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    total_time = time.time() - whole_time
    processTime(total_time, "Whole time to process the algorithm")
