#!/usr/bin/env python3

"""
@author: omadz

Based on the following tutorial :
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocessing as pp
import logging
# from text_cnn import TextCNN
from tensorflow.contrib import learn


if __name__ == '__main__':

    # Logger
    # ==================================================
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler which logs even debug messages
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)

    # Console handler which logs info messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s -- %(name)s -- %(levelname)s\n'
                                  '%(message)s\n')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(" ========== Début du programme ========== ")

    # Parameters
    # ==================================================

    # Data loading parameters
    tf.flags.DEFINE_float(
            "dev_sample_percentage", .1,
            "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string(
            "positive_data_file", "../data/rt-polaritydata/rt-polarity.pos",
            "Data source for the positive data.")
    tf.flags.DEFINE_string(
            "negative_data_file", "../data/rt-polaritydata/rt-polarity.neg",
            "Data source for the negative data.")

    # Model Hyperparameters
    tf.flags.DEFINE_integer(
            "embedding_dim", 128,
            "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string(
            "filter_sizes", "3,4,5",
            "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer(
            "num_filters", 128,
            "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float(
            "dropout_keep_prob", 0.5,
            "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float(
            "l2_reg_lambda", 0.0,
            "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer(
            "batch_size", 64,
            "Batch Size (default: 64)")
    tf.flags.DEFINE_integer(
            "num_epochs", 200,
            "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer(
            "evaluate_every", 100,
            "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer(
            "checkpoint_every", 100,
            "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer(
            "num_checkpoints", 5,
            "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    tf.flags.DEFINE_boolean(
            "allow_soft_placement", True,
            "Allow device soft device placement")
    tf.flags.DEFINE_boolean(
            "log_device_placement", False,
            "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    logger.debug(" *** Parameters *** ")
    for attr, value in sorted(FLAGS.__flags.items()):
        logger.debug("%s=%s", attr.upper(), value)
    logger.debug(" *** END - Parameters *** ")

    # Data Preparation
    # ==================================================

    # Load data
    logger.info(" *** Loading data *** ")
    x_text, y = pp.load_data_and_labels(FLAGS.positive_data_file,
                                        FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    logger.debug("Max document length : %s", max_document_length)
    vocab_processor = (learn.preprocessing.VocabularyProcessor
                       (max_document_length))
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    logger.debug("Data (shape : %s):\n %s", x.shape, x)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    logger.debug("Data shuffled (shape: %s):\n %s", x_shuffled.shape,
                 x_shuffled)
    y_shuffled = y[shuffle_indices]
    logger.debug("Label shuffled (shape: %s):\n %s", y_shuffled.shape,
                 y_shuffled)

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = (x_shuffled[:dev_sample_index],
                      x_shuffled[dev_sample_index:])
    y_train, y_dev = (y_shuffled[:dev_sample_index],
                      y_shuffled[dev_sample_index:])

    logger.info(" ==> VOCABULARY <== ")
    logger.info("Vocabulary size : %s", len(vocab_processor.vocabulary_))
    logger.debug("Log the first 10 words of the vocabulary and the last one")
    # Log the first 10 words and the final one
    for i in range(0, len(vocab_processor.vocabulary_)):
        logger.debug("Word in the vocabulary : %s",
                     vocab_processor.vocabulary_.reverse(i))
        if (i == 10):
            break
    logger.debug("Last word in the vocabulary : %s",
                 vocab_processor.vocabulary_.reverse(
                         len(vocab_processor.vocabulary_) - 1))
    logger.info(" ==> VOCABULARY <== ")

    logger.info("Train/Dev : %s/%s", len(y_train), len(y_dev))

    logger.debug(" *** END - Loading data *** ")

    logger.info(" ========== Fin du programme ========== ")
