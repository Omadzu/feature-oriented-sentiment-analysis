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
# BeautifoulSoup
from bs4 import BeautifulSoup as bs
import re
# Download text data sets, including stop words
from nltk.corpus import stopwords
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
import multiprocessing as mp
import time
# Random forest
from sklearn.ensemble import RandomForestClassifier

""" Methods """
def review_to_words( raw_review ):
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
    return ( " ".join(meaningful_words) )
    
def calculateParallel(listOfReviews, threads= mp.cpu_count()):
    # see : https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
    # see : http://stackoverflow.com/questions/32789086/using-pool-map-to-apply-function-to-list-of-strings-in-parallel
    
    # Initialize an empty list to hold the clean reviews
    clean_reviews = []
    
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
        
if __name__ == '__main__':
    # Collect the data
    train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)
    
    # Get the number of the reviews based on the dataframe column size
    num_reviews = train["review"].size
    
    print "Cleaning and parsing the training set movie reviews...\n"
    
    # Distribute on the processor cores the calculation
    mapResults = calculateParallel(train["review"])
    processProgress(mapResults)
    clean_train_reviews = mapResults.get()

    print "Creating bag of words...\n"
    
    # Initialize the CountVectorizer object, which is scikit-learn's
    # bag of words tool
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)
    
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
    #print vocab
    
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
#    for tag, count in zip(vocab, dist):
#        print count, tag

    print "Training the random forest..."
    
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)
    
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_data_features, train["sentiment"])
    
    # =========================
    
    # Read the test data
    test = pd.read_csv("../../data/testData.tsv", header=0, delimiter="\t", \
                       quoting=3)
    
    # Verify that there are 25,000 rows and 2 columns
    print test.shape
    
    print "Cleaning and parsing the test set movie reviews...\n"
    
    # Distribute on the processor cores the calculation
    mapResults = calculateParallel(test["review"])
    processProgress(mapResults)
    clean_test_reviews = mapResults.get()
    
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    # Use the random forest to make sentimental label predictions
    result = forest.predict(test_data_features)
    
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    
    # Use pandas to write the comma-separated output file
    output.to_csv( "results/Bag_of_Words_model.csv", index=False, quoting=3 )
    
    print "Prediction complete on test set !"