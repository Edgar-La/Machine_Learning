# -*- coding: utf-8 -*-
"""Confusion_Matrix.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Edgar-La/Machine_Learning/blob/main/Confusion_Matrix/Confusion_Matrix.ipynb

__Author:__ Edgar Lara

__Date:__ March 6, 2021

# __Implementing confusion matrix__
"""

import sklearn.metrics as skl
import numpy as np
import matplotlib.pyplot as plt
import itertools



def ACC(confusion_matrix, labels_2):
    diag = 0
    for i in range(len(confusion_matrix)):
      diag += confusion_matrix[i][i]
    return diag/len(labels_2)

#Help from https://stackoverflow.com/questions/40264763/how-can-i-make-my-confusion-matrix-plot-only-1-decimal-in-python
def plot_confusion_matrix(cm, ACC,title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title + '. ACC: ' + str(ACC))
    title = title + '. ACC: ' + str(ACC)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()