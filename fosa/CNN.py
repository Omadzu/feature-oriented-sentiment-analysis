#!/usr/bin/env python3

"""
Implementation of the CNN algorithm used for polarity learning and prediction.
"""

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.

    Uses an embedding layer, followed by a convolutional,
    max-pooling and softmax layer.
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters):
        """
        :param sequence_length: Length of the sentences. Here, all sentences\
        have the same length because of the padding of the preprocessing part.
        :type sequence_length: int

        :param num_classes: Number of classes in the output layer, two in this\
        case (positive and negative).
        :type num_classes: int

        :param vocab_size: The size of the vocabulary. This is needed to\
        define the size of the embedding layer, which will have shape\
        [vocabulary_size, embedding_size].
        :type vocab_size: int

        :param embedding_size: The dimensionality of our embeddings.
        :type embedding_size: int

        :param filter_sizes: The number of words we want our convolutional\
        filters to cover. We will have num_filters for each size specified\
        here. For example, [3, 4, 5] means that we will have filters that\
        slide over 3, 4 and 5 words respectively, for a total of\
        3 * num_filters filters.
        :type filter_sizes: array

        :param num_filters: The number of filters per filter size (see above).
        :type num_filters: int

        .. todo::
            Verify types.

        .. todo::
            Modify the CNN to adapt it to SemEval dataset.
        """

        # Placeholders for input, output and dropout
        # ==================================================

        self.input_x = tf.placeholder(tf.int32,
                                      [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # Operations can be executed in CPU or GPU (default)
        # TODO : Test CPU and GPU mode
        # -
        # Embedding layer
        # Good explanation : https://stackoverflow.com/questions/37897934/tensorflow-embedding-lookup
        # ==================================================

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Weights - filter matrix
            W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")

            # Compute embedding (array of size embedding_size) for input_x
            # Output shape : [None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

            # Expend dimension of embedded_chars to feed it later to CNN
            # Output shape : [None, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,
                                                          -1)

        # Create a convolution + maxpool layer for each filter size
        # ==================================================

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                # ----------------

                filter_shape = [filter_size, embedding_size, 1, num_filters]

                # Weights - filter matrix
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")

                # Biases
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Max-pooling over the outputs
                # Output shape : [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        # Shape : [batch_size, num_filters_total]
        # ==================================================

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # ==================================================

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        self.dropout_keep_prob)

        # Scores and predictions
        # ==================================================

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(
                    [num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        # ==================================================

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,
                                                             self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        # ==================================================

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"), name="accuracy")
