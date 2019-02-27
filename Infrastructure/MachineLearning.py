from Infrastructure import ImportData, EditData
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def GetSGDClassifier2(X_train, X_test, y_train, y_test):
    n = 36000
    some_digit_image = X_train[n]
    some_digit_target = y_train[n]

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_classificator = SGDClassifier(random_state=42, max_iter=5, tol=-np.inf)
    sgd_classificator.fit(X_train, y_train_5)
    score = cross_val_score(sgd_classificator, X_train, y_train_5, cv=3, scoring="accuracy")
    y_train_predict = cross_val_predict(sgd_classificator, X_train, y_train_5, cv=3)

    confusion_matrix_digits = confusion_matrix(y_train_5, y_train_predict)

    precision_s = precision_score(y_train_5, y_train_predict)
    pelnosc_recall = recall_score(y_train_5, y_train_predict)
    f1 = f1_score(y_train_5, y_train_predict)

    y_scores = sgd_classificator.decision_function([some_digit_image])
    threshold_df = 200000
    y_some_digit_pred = (y_scores > threshold_df)

    y_scores_plot = cross_val_predict(sgd_classificator, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_plot)
    # EditData.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    # EditData.plot_precision_vs_recall(precisions, recalls)
    p = 70000
    y_train_pred_90 = (y_scores_plot > p)

    #EditData.plot_roc_curve(y_train_5, y_scores_plot)
    if False:
        print("predykcja: ", sgd_classificator.predict([some_digit_image]))
        print("target: ", some_digit_target)
        print("Score: ", score)
        print("confusion matrix: [PN, FP]: ", confusion_matrix_digits[0], ", [FN, PP]: ", confusion_matrix_digits[1])

        print("Wynik F1: ", f1)
        print("predykcja z progiem ", threshold_df, ": ", y_some_digit_pred)
        print("Precyzja z progiem", p, precision_score(y_train_5, y_train_pred_90))
        print("Pełność z progiem", p, recall_score(y_train_5, y_train_pred_90))
        print("................................................................................")
        print("ROC")
    print("........SGD............")
    print("Precyzja SGD: ", np.round(precision_s, 4))
    print("Pełność SGD: ", np.round(pelnosc_recall, 4))
    print(".......................")
    return y_train_5, y_scores_plot


def Get10SGDClassifiers(X_train, X_test, y_train, y_test):
    sgd_classificator = SGDClassifier(random_state=42, max_iter=5, tol=-np.inf)
    sgd_classificator.fit(X_train, y_train)
    predict = sgd_classificator.predict([X_test[1]])
    array_score = sgd_classificator.decision_function([X_test[1]])
    print("każda cyfra ma swój klasyfikator")
    print("predykcja: ", predict)
    print("target: ", y_test[1])
    print("klasy: ", sgd_classificator.classes_)
    print("macierz punktów: ", array_score)


def GetSGDClassifier(X_train, X_test, y_train, y_test):
    y_train_5_SGD = (y_train == 5)
    y_test_5_SGD = (y_test == 5)
    sgd_classificator = SGDClassifier(random_state=42, max_iter=5, tol=-np.inf)
    y_train_predict = cross_val_predict(sgd_classificator, X_train, y_train_5_SGD, cv=3)
    precision = precision_score(y_train_5_SGD, y_train_predict)
    pelnosc = recall_score(y_train_5_SGD, y_train_predict)
    y_scores_SGD = cross_val_predict(sgd_classificator, X_train, y_train_5_SGD, cv=3, method="decision_function")
    print("...........SGD...............")
    print("Precyzja SGD: ", np.round(precision, 4))
    print("Pełność SGD: ", np.round(pelnosc, 4))
    print(".............................")
    return y_train_5_SGD, y_scores_SGD


def GetForestClassifier(X_train, X_test, y_train, y_test):
    #predict_proba - lista prawdopodobieńst
    y_train_5_forest = (y_train == 5)
    y_test_5_forest = (y_test == 5)
    randomForestClassifier = RandomForestClassifier(random_state=42, n_estimators=10)
    y_probas_forest = cross_val_predict(randomForestClassifier, X_train, y_train_5_forest, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    y_train_predict = cross_val_predict(randomForestClassifier, X_train, y_train_5_forest, cv=3)
    precision = precision_score(y_train_5_forest, y_train_predict)
    pelnosc = recall_score(y_train_5_forest, y_train_predict)
    print("......Random Forest..........")
    print("Precyzja Random Forest: ", np.round(precision, 4))
    print("Pełność Random Forest: ", np.round(pelnosc, 4))
    print(".............................")
    return y_train_5_forest, y_scores_forest







def GetOVOSGDClassirfier(X_train, X_test, y_train, y_test):
    sgdClassifier = SGDClassifier(random_state=42, max_iter=5, tol=-np.inf)
    ovoClassifier = OneVsOneClassifier(sgdClassifier)
    ovoClassifier.fit(X_train, y_train)
    predict = ovoClassifier.predict([X_test[1]])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    dokladnosc = cross_val_score(sgdClassifier, X_train, y_train, cv=3, scoring="accuracy")
    dokladnosc_scale = cross_val_score(sgdClassifier, X_train_scaled, y_train, cv=3, scoring="accuracy")

    y_train_pred = cross_val_predict(sgdClassifier, X_train_scaled, y_train, cv=3)
    matrix_conf = confusion_matrix(y_train, y_train_pred)
    row_sums = matrix_conf.sum(axis=1, keepdims=True)
    norm_conf_matrix = matrix_conf/row_sums
    np.fill_diagonal(norm_conf_matrix, 0)

    plt.matshow(norm_conf_matrix, cmap=plt.cm.gray)



    print("każda cyfra ma swój klasyfikator")
    print("predykcja: ", predict)
    print("target: ", y_test[1])
    print("dokladnosc: ", dokladnosc)
    print("dokladnosc po skalowaniu: ", dokladnosc_scale)
    print("macierz pomyłek: ", matrix_conf)


def GetForestClassifier2(X_train, X_test, y_train, y_test):
    #predict_proba - lista prawdopodobieńst
    randomForestClassifier = RandomForestClassifier(random_state=42, n_estimators=10)
    randomForestClassifier.fit(X_train, y_train)
    n = 1591
    predict = randomForestClassifier.predict([X_test[n]])

    print("......Random Forest..........")
    print("Predykcja: ", predict)
    print("Target: ", y_test[n])
    print("Prawdopodobiensta: ", randomForestClassifier.predict_proba([X_test[n]]))
    print(".............................")


def GetMultiLabelKNeiborsClassifier(X_train, X_test, y_train, y_test):
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 0)
    y_multilabrl = np.c_[y_train_large, y_train_odd]

    kneighbors = KNeighborsClassifier()
    #kneighbors.fit(X_train, y_multilabrl)

    y_train_knn_pred = cross_val_predict(kneighbors, X_train, y_multilabrl, cv=3)
    f1 = f1_score(y_multilabrl, y_train_knn_pred, average="macro")

    print(kneighbors.predict([X_test[1]]))
    print("target: ", y_test[1])
    print(f1)
