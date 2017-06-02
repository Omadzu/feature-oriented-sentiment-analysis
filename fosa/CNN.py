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
        # Implementation...
