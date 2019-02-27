from Infrastructure import ImportData, EditData, MachineLearning
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.multiclass import OneVsOneClassifier

print("----------------START---------------------")

X, y = ImportData.get_mnist_data()
X_train, X_test, y_train, y_test = EditData.GetTrainAndTestData(X, y)

# y_train_5_SGD, y_scores_SGD = MachineLearning.GetSGDClassifier(X_train, X_test, y_train, y_test)
# y_train_5_forest, y_scores_forest = MachineLearning.GetForestClassifier(X_train, X_test, y_train, y_test)
# EditData.plot_roc_curve(y_train_5_SGD, y_scores_SGD, "SGD:")
# EditData.plot_roc_curve(y_train_5_forest, y_scores_forest, "Forest:")


MachineLearning.GetMultiLabelKNeiborsClassifier(X_train, X_test, y_train, y_test)

#MachineLearning.GetForestClassifier2(X_train, X_test, y_train, y_test)

plt.show()
print("----------------END-----------------------")
