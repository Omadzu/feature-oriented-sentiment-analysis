#! /usr/bin/env python3

"""
Prediction part of the algorithm.
"""

import tensorflow as tf
import numpy as np
import os
import preprocessing as pp
from tensorflow.contrib import learn
import csv
from sklearn import metrics
from pandas_ml import ConfusionMatrix
import yaml
import logging

# Constants
# ============================================

SEMEVAL_FOLDER = '../data/SemEval/Subtask1'
RESTAURANT_TRAIN = os.path.join(SEMEVAL_FOLDER, 'restaurant', 'train.xml')
RESTAURANT_TEST = os.path.join(SEMEVAL_FOLDER, 'restaurant', 'test',
                               'test_gold.xml')
LAPTOP_TRAIN = os.path.join(SEMEVAL_FOLDER, 'laptop', 'train.xml')
LAPTOP_TEST = os.path.join(SEMEVAL_FOLDER, 'laptop', 'test', 'test_gold.xml')

# Functions
# ==================================================


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

if __name__ == '__main__':

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Parameters
    # ==================================================

    # Data Parameters

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "",
                           "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", False,
                            "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True,
                            "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False,
                            "Log placement of ops on devices")

    # Precise if predictions is on features or polarity
    tf.flags.DEFINE_string("focus", "", "'feature' or 'polarity'")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    # Logger
    # ==================================================

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler which logs even debug messages
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)

    # Other file handler to store information for each run
    log_directory = os.path.join(FLAGS.checkpoint_dir, "..", "eval.log")
    run_file_handler = logging.FileHandler(log_directory)
    run_file_handler.setLevel(logging.DEBUG)

    # Console handler which logs info messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    run_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(run_file_handler)
    logger.addHandler(console_handler)
    logger.debug(" *** Parameters *** ")
    for attr, value in sorted(FLAGS.__flags.items()):
        logger.debug("{}={}".format(attr.upper(), value))
    logger.debug("")

    datasets = None

    # CHANGE THIS: Load data. Load your own data here
    dataset_name = cfg["datasets"]["default"]
    if FLAGS.eval_train:
        if dataset_name == "mrpolarity":
            datasets = pp.get_datasets_mrpolarity(
                cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                cfg["datasets"][dataset_name]["negative_data_file"]["path"])
        elif dataset_name == "20newsgroup":
            datasets = pp.get_datasets_20newsgroup(
                subset="test",
                categories=cfg["datasets"][dataset_name]["categories"],
                shuffle=cfg["datasets"][dataset_name]["shuffle"],
                random_state=cfg["datasets"][dataset_name]["random_state"])
        elif dataset_name == "localdata":
            datasets = pp.get_datasets_localdata(
                container_path=cfg["datasets"][dataset_name]["test_path"],
                categories=cfg["datasets"][dataset_name]["categories"],
                shuffle=cfg["datasets"][dataset_name]["shuffle"],
                random_state=cfg["datasets"][dataset_name]["random_state"])
        elif dataset_name == "semeval":
            current_domain = cfg["datasets"][dataset_name]["current_domain"]
            focus = FLAGS.focus
            if current_domain == 'RESTAURANT':
                datasets = pp.get_dataset_semeval(RESTAURANT_TEST, focus)
            elif current_domain == 'LAPTOP':
                datasets = pp.get_dataset_semeval(LAPTOP_TEST, focus)
            else:
                raise ValueError("The 'current_domain' parameter in the " +
                                 "'config.yml' file must be 'RESTAURANT' " +
                                 "or 'LAPTOP'")

        x_raw, y_test = pp.load_data_and_labels(datasets)
        y_test = np.argmax(y_test, axis=1)
        logger.debug("Total number of test examples: {}".format(len(y_test)))
    else:
        if dataset_name == "mrpolarity":
            datasets = {"target_names": ['positive_examples',
                                         'negative_examples']}
            x_raw = ["a masterpiece four years in the making",
                     "everything is off."]
            y_test = [1, 0]
        elif dataset_name == "20newsgroup":
            datasets = {"target_names": ['alt.atheism',
                                         'comp.graphics',
                                         'sci.med',
                                         'soc.religion.christian']}
            x_raw = ["The number of reported cases of gonorrhea in Colorado" +
                     "increased",
                     "I am in the market for a 24-bit graphics card for a PC"]
            y_test = [2, 1]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
            vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    logger.info("")
    logger.info("Evaluation :")
    logger.info("")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(
                    "{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name(
                    "dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name(
                    "output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = pp.batch_iter(
                    list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            for x_test_batch in batches:
                batch_predictions_scores = sess.run(
                        [predictions, scores],
                        {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate(
                        [all_predictions, batch_predictions_scores[0]])
                probabilities = softmax(batch_predictions_scores[1])
                if all_probabilities is not None:
                    all_probabilities = np.concatenate(
                            [all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        logger.debug("Total number of test examples: {}".format(len(y_test)))
        logger.info("")
        logger.info(
                "Accuracy: {:g}".format(
                        correct_predictions/float(len(y_test))))

        logger.info(metrics.classification_report(
                y_test, all_predictions,
                target_names=datasets['target_names']))

        confusion_matrix = ConfusionMatrix(y_test, all_predictions)
        logger.info(confusion_matrix)
        logger.info("")
        str_labels = "Labels : "
        for idx, label in enumerate(datasets['target_names']):
            str_labels += "{} = {}, ".format(idx, label)
        logger.info(str_labels)
        logger.info("")

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack(
            (np.array(x_raw),
             [int(prediction) for prediction in all_predictions],
             ["{}".format(probability) for probability in all_probabilities]))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")

    logger.info("Saving evaluation to {0}".format(out_path))

    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
