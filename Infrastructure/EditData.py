import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score


def GetTrainAndTestData(X, y):
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y[shuffle_index]
    return X_train, X_test, y_train, y_test

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precyzja")
    plt.plot(thresholds, recalls[:-1], "g-", label="Pełność")
    plt.xlabel("Próg")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "g-", label="Precyzja")
    plt.xlabel("Pełność")
    plt.ylabel("Precyzja")
    plt.show()


def plot_roc_curve(y_train, y_scores, title=""):
    fpr, tpr, thresholds_roc = roc_curve(y_train, y_scores)
    lab = title + " AUC: {0}".format(np.round(roc_auc_score(y_train, y_scores), 4))
    plt.plot(fpr, tpr, label=lab)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Odsetek fałszywie pozytywnych')
    plt.ylabel('Odsetek prawdziwie pozytywnych')
    plt.legend(loc="center left")

