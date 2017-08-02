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

# Constants
# ==================================================

COLORS = [(255, 105, 180),
          (233, 150, 122),
          (220, 20, 60),
          (139, 0, 0),
          (255, 69, 0),
          (255, 140, 0),
          (255, 215, 0),
          (210, 105, 30),
          (165, 42, 42),
          (128, 128, 0),
          (46, 139, 87),
          (0, 100, 0),
          (0, 255, 255),
          (95, 158, 160),
          (25, 25, 112),
          (218, 112, 214),
          (75, 0, 130)]

# Definitions
# ==================================================


def convert_to_rgb_matplot(rgb_tuple):
    red, green, blue = rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]

    def operation(color):
        return float(color/255)

    red = operation(red)
    green = operation(green)
    blue = operation(blue)

    return (red, green, blue)


def convert_constant_color():
    scaled_colors = []
    for color in COLORS:
        scaled_colors.append(convert_to_rgb_matplot(color))

    return scaled_colors


def lighter(color, percent):
    """
    The 'color' tuple must be between (0, 0, 0) and (255, 255, 255)
    From : https://stackoverflow.com/questions/28015400/how-to-fade-color
    """
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white - color
    return color + vector * percent


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
    colors = convert_constant_color()

    fig1, ax1 = plt.subplots()
    patches, texts, _ = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                startangle=90, colors=colors)
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
        bar_i = ax.bar(ind + i * bar_width, toPlot[i], bar_width,
                       color=convert_to_rgb_matplot(COLORS[i]))
        bars.append(bar_i)

    # Add some text for lavels, title and axes ticks
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


if __name__ == '__main__':

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Parameters
    # ==================================================

    # Data Parameters

    # Eval Parameters
    tf.flags.DEFINE_string("predictions_csv", "",
                           "CSV file where predictions are stored")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    prediction_file = pd.read_csv(FLAGS.predictions_csv)

    # Correct predictions
    # ==================================================

    number_pred = prediction_file.text.size

    # Feature
    row_ids_same = prediction_file[
            prediction_file.feature == prediction_file.pred_feature].index
    number_same = row_ids_same.size
    perc_correct_feature = number_same/number_pred*100

    # Polarity
    row_ids_same = prediction_file[
            prediction_file.polarity == prediction_file.pred_polarity].index
    number_same = row_ids_same.size
    perc_correct_polarity = number_same/number_pred*100

    # Whole prediction
    row_ids_same = prediction_file[
            prediction_file.new_class == prediction_file.pred_new_class].index
    number_same = row_ids_same.size
    perc_correct_whole = number_same/number_pred*100

    # Display pie charts
    # ==================================================

    # Data to plot
    labels = 'Correct features', 'Incorrect features'
    sizes = [perc_correct_feature, 100 - perc_correct_feature]
    colors = ['gold', 'red']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

    # Data to plot
    labels = 'Correct polarities', 'Incorrect polarities'
    sizes = [perc_correct_polarity, 100 - perc_correct_polarity]
    colors = ['gold', 'red']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

    # Data to plot
    labels = 'Correct predictions', 'Incorrect predictions'
    sizes = [perc_correct_whole, 100 - perc_correct_whole]
    colors = ['gold', 'red']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()
