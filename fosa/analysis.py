#!/usr/bin/env python3
"""
Display some information like pie charts...etc... in order to analyse the
results of the evaluation part of the algorithm.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
import preprocessing as pp

# Constants
# ==================================================

# Definitions
# ==================================================


def pie_chart_support_distribution(classification_report, title, folder):
    """
    Plot a pie chart which describes the distribution of each class.
    :param classification_report: Sliced classification report : classes,
    toPlot, support. toPlot must be a tuple (precision, recall, f1-score)
    """

    classes, toPlot, support = slice_classification_report(
            classification_report)

    # Don't take into account the last column which is the total number
    # of each class
    labels = classes[0:len(classes)-1]
    sizes = support[0:len(classes)-1]

    fig1, ax1 = plt.subplots()
    patches, texts, _ = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    ax1.set_title(title)
    ax1.legend(patches, labels, loc="best")

    plt.savefig(folder+"/"+title.replace(" ", "_")+".png", format="png",
                dpi=1000)


def bar_chart_classification_report(classification_report, title, folder):
    """
    Plot a bar graph which sums up the classification report of the scikit
    learn tool.
    :param classification_report: Sliced classification report : classes,
    toPlot, support. toPlot must be a tuple (precision, recall, f1-score)
    """
    classes, toPlot, support = slice_classification_report(
            classification_report)

    N = 3
    bar_width = 0.05

    ind = np.arange(N)
    fig, ax = plt.subplots()

    # Enumerate over each class except the last one which represent the average
    # and total
    bars = []
    for i in range(len(classes)):
        bar_i = ax.bar(ind + i * bar_width, toPlot[i], bar_width)
        bars.append(bar_i)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.set_xticks(ind + bar_width / len(classes))
    ax.set_xticklabels(("Precision", "Recall", "F1-score"))

    ax.legend(bars, classes, loc="best")

    plt.savefig(folder+"/"+title.replace(" ", "_")+".png", format="png",
                dpi=1000)


def slice_classification_report(classification_report):
    """
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    """
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []

    for line in lines[2: (len(lines) - 2)]:
        t = line.strip().split()

        if len(t) < 2:
            continue

        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    # Save the average precision/recall/F1-score and total support
    t = lines[len(lines) - 2].strip().split()
    classes.append(t[0] + t[1] + t[2])
    v = [float(x) for x in t[3: len(t) - 1]]
    support.append(int(t[-1]))
    class_names.append(t[0] + t[1] + t[2])
    plotMat.append(v)

    print("\n")
    print("plotMat: {0}".format(plotMat))
    print("support: {0}".format(support))

    return classes, plotMat, support


def display_stat(filepath):
    """
    Statistics from the SemEval 2016 competition, Task 5, Subtask 1 dataset.
    :param filepath: Path of the dataset SemEval. The path must leads to a
    folder containing both the training and testing sets.
    :type filepath: string
    :return: Pandas.dataframe with the following columns : review_id,
    sentence_id, text, feature, polarity
    """

    training_set = pp.parse_XML(filepath+"/train.xml")
    testing_set = pp.parse_XML(filepath+"/test/test_gold.xml")

    # Some opinions concerns various food, drinks...etc... but the opinion
    # is the same while the target differ. So deleting duplicates as the scope
    # of this study does not imply target (OPE in SemEval)
    training_set = training_set.drop_duplicates()
    testing_set = testing_set.drop_duplicates()

    # Count # of opinions for each sentence
    count_opinions_train = training_set['sentence_id'].value_counts()
    count_opinions_train = count_opinions_train.value_counts()
    count_opinions_test = testing_set['sentence_id'].value_counts()
    count_opinions_test = count_opinions_test.value_counts()

    # Display pie charts
    count_dict_train = count_opinions_train.to_dict()
    labels = list(count_dict_train.keys())
    sizes = list(count_dict_train.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Percentage of opinion occurences in a sentence')

    count_dict_test = count_opinions_test.to_dict()
    labels = list(count_dict_test.keys())
    sizes = list(count_dict_test.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Percentage of opinion occurences in a sentence')

    plt.show()


if __name__ == '__main__':

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Parameters
    # ==================================================

    # Data Parameters

    # Eval Parameters
    tf.flags.DEFINE_boolean("display_stat", True,
                            "Display statistics of SemEval dataset")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    dataset_filepath = "../data/SemEval/Subtask1/restaurant"

    if FLAGS.display_stat:
        display_stat(dataset_filepath)
