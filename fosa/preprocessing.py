#!/usr/bin/env python3

"""
Tools for the preprocessing part of the algorithm.
"""

import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import codecs

# Constants
# ============================================

SEMEVAL_FOLDER = '../data/SemEval/Subtask1'
RESTAURANT_TRAIN = os.path.join(SEMEVAL_FOLDER, 'restaurant', 'train.xml')
RESTAURANT_TEST = os.path.join(SEMEVAL_FOLDER, 'restaurant', 'test',
                               'test_gold.xml')
LAPTOP_TRAIN = os.path.join(SEMEVAL_FOLDER, 'laptop', 'train.xml')
LAPTOP_TEST = os.path.join(SEMEVAL_FOLDER, 'laptop', 'test', 'test_gold.xml')
RESTAURANT_ENTITIES = ['FOOD', 'DRINKS', 'SERVICE', 'RESTAURANT', 'AMBIENCE',
                       'LOCATION']
LAPTOP_ENTITIES = ['LAPTOP', 'HARDWARE', 'SHIPPING', 'COMPANY', 'SUPPORT',
                   'SOFTWARE']
# The following entities will be simplified as HARDWARE entity
HARDWARE = ['DISPLAY', 'CPU', 'MOTHERBOARD', 'HARD_DISC', 'MEMORY', 'BATTERY',
            'POWER_SUPPLY', 'KEYBOARD', 'MOUSE', 'FANS_COOLING',
            'OPTICAL_DRIVES', 'PORTS', 'GRAPHICS', 'MULTIMEDIA_DEVICES']
POLARITY = ['positive', 'neutral', 'negative']


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def batch_number(data, batch_size, num_epochs):
    """
    Compute the number of batch to process during the epoch loop
    """

    return int((len(data)-1)/batch_size) + 1


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = batch_number(data, batch_size, num_epochs)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True,
                             random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories,
                                  shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates
    labels. Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None,
                           load_content=True, encoding='utf-8', shuffle=True,
                           random_state=42, decode_error='replace'):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder
    structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by
    default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path,
                          categories=categories,
                          load_content=load_content,
                          shuffle=shuffle, encoding=encoding,
                          random_state=random_state,
                          decode_error=decode_error)

    return datasets


def load_data_and_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """

    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # Load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())

        # Initial matrix with random uniform
        embedding_vectors = np.random.uniform(
                -0.25, 0.25, (len(vocabulary), vector_size))

        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; " +
                                       "is count incorrect or file " +
                                       "otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(
                            f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                        raise EOFError("unexpected end of input; " +
                                       "is count incorrect or file " +
                                       "otherwise damaged?")
                parts = str(line.rstrip(),
                            encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s" +
                                     "(is this really the text format?)"
                                     % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # Load embedding_vectors from the glove
    # Initial matrix with random uniform
    embedding_vectors = np.random.uniform(
            -0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def get_dataset_semeval(filepath=RESTAURANT_TRAIN, focus='polarity'):
    """
    Parse the XML document of SemEval competition (SemEval 2016, Task 5,
    Subtask 1). The targetted domains are those containing English reviews :
    the laptop and the restaurant domains.

    During this step, the XML file is parsed in a Pandas.DataFrame in order to
    easily manipulate the SemEval data.

    There are different outputs for this function depending on what the user
    wants to focus on. The ouput can focus on the features or on the polarity
    depending on the 'focus' parameter.

    :param filepath: Default : training dataset of the restaurant domain. Path
    of the dataset SemEval.
    :type filepath: string
    :param focus: (required) Default : polarity. Possible choices :'feature',
    'polarity'. Throw an error if not specified. If 'feature' is specified,
    each sentence will be assigned one or multiple features depending on which
    feature the sentence is about. If 'polarity' is specified, each sentence
    will be assigned one or multiple polarities depending on which polarities
    are expressed in the sentence.
    :type focus: string
    :return: A dictionnary representing the SemEval dataset. This dictionnary
    will be focusing on either the features or the polarities included in the
    sentences.
    """
    # TODO : Multilabel for feature and polarity.
    # TODO : CURRENT restaurant, adapt to also do auto LAPTOP

    with codecs.open(filepath, 'r', 'utf8') as xml_file:
        xml_tree = ET.parse(xml_file)
        root = xml_tree.getroot()

        list_for_dataframe = []
        reviews = root.findall('Review')
        for review in reviews:

            review_id = review.get('rid')

            # Extract opinions and text for each sentence
            # ATTENTION : Only some entities are taking into account for now.
            # So split the category until # and select only some entities
            # ============================================

            sentences = review.findall('./sentences/sentence')

            for sentence in sentences:
                sentence_id = sentence.get('id')
                text = sentence.find('text').text
                opinions = sentence.findall('./Opinions/Opinion')

                for opinion in opinions:
                    category = opinion.get('category')
                    category = category.split('#')[0]
                    polarity = opinion.get('polarity')

                    list_for_dataframe.append({
                            'review_id': review_id,
                            'sentence_id': sentence_id,
                            'text': text,
                            'feature': category,
                            'polarity': polarity})

        dataset_df = pd.DataFrame(list_for_dataframe)

        # The dataset is composed of either features or polarity in order to
        # be feed to the CNN.
        #
        # Some other manipulations are done on the data :
        # - rename some features for simplicity (see HARDWARE constant)
        # - suppress the sentences which has features not studied in this
        #   scope
        # - remove duplicates
        #
        # A dictionary representing the dataset is built :
        # - datasets['data'] = x
        # - datasets['target'] = y
        # - datasets['target_names'] = possible categories
        # The values stored in datasets['target'] are integers in order to
        # be read further in the algorithm.
        # ============================================

        datasets = {}

        if focus == 'polarity':
            dataset_df = dataset_df[['text', 'polarity']]
            dataset_df = dataset_df.rename(columns={'polarity': 'y'})

            range_dict = list(range(len(POLARITY)))
            polarity_dict = dict(zip(POLARITY, range_dict))
            dataset_df = dataset_df.replace({'y': polarity_dict})
            datasets['target_names'] = POLARITY
        elif focus == 'feature':
            dataset_df = dataset_df[['text', 'feature']]
            dataset_df = dataset_df.rename(columns={'feature': 'y'})

            dataset_df = dataset_df.replace(HARDWARE, 'HARDWARE')

            if filepath == RESTAURANT_TRAIN or filepath == RESTAURANT_TEST:
                datasets['target_names'] = RESTAURANT_ENTITIES
                dataset_df = dataset_df[dataset_df['y'].isin(RESTAURANT_ENTITIES)]

                range_dict = list(range(len(RESTAURANT_ENTITIES)))
                restaurant_entities_dict = dict(zip(RESTAURANT_ENTITIES,
                                                    range_dict))
                dataset_df = dataset_df.replace({'y':
                                                restaurant_entities_dict})
            elif filepath == LAPTOP_TRAIN or filepath == LAPTOP_TEST:
                datasets['target_names'] = LAPTOP_ENTITIES
                dataset_df = dataset_df[dataset_df['y'].isin(LAPTOP_ENTITIES)]

                range_dict = list(range(len(LAPTOP_ENTITIES)))
                laptop_entities_dict = dict(zip(LAPTOP_ENTITIES, range_dict))
                dataset_df = dataset_df.replace({'y': laptop_entities_dict})
            else:
                raise ValueError("'filepath' parameter must use the " +
                                 "following constants : 'RESTAURANT_TRAIN', " +
                                 "'RESTAURANT_TEST', 'LAPTOP_TRAIN', " +
                                 "'LAPTOP_TEST'")

        else:
            raise ValueError("'focus' parameter must be 'feature' or" +
                             "'polarity'")

        dataset_df = dataset_df.drop_duplicates()

        datasets['data'] = dataset_df['text'].values.tolist()
        datasets['target'] = dataset_df['y'].values.tolist()

        return datasets
