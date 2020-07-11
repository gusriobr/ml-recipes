import math

import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.linear_model as skmod
import sklearn.metrics as skmet
import sklearn.model_selection as skmodsel
import sklearn.svm as sksvm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import recipes.evaluation_classification as evalc
import recipes.exploratory_analysis as eda
import recipes.feature_reduction as freduct
import recipes.feature_selection as fselect
from recipes.exploratory_analysis import descriptive_statistics, print_column_datatypes
from recipes.preprocessing import polynomial_features


# Python Project Template


def config_printing():
    np.set_printoptions(linewidth=500)
    np.set_printoptions(precision=4)
    pd.set_option('display.width', 500)
    pd.set_option('precision', 4)


def read_data():
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv("resources/housing.csv", delim_whitespace=True, names=names)
    return df


def feature_generation(df, excluded_columns=None):
    df = polynomial_features(df, excluded_columns)
    return df


def treat_missing_data(df, categorical_cols):
    # 1 - Mark wrong values as NAN example
    # df[[1, 2, 3, 4, 5]] = df[[1, 2, 3, 4, 5]].replace(0, numpy.NaN)
    # 2 - Drop missing values
    # 3 - Imputer
    # replace missing data using column average value
    df.fillna(df.mean(), inplace=True)
    return df


if __name__ == '__main__':
    # 1. Prepare Problem
    classification_tasks = False

    exploratory_analysis = False
    feature_selection = True
    model_evaluation = True

    # b) Load dataset
    df = read_data()

    # 2. Summarize Data
    # a) Descriptive statistics
    # b) Data visualizations

    # Data analysis
    df = read_data()
    print_column_datatypes(df)
    # descriptive_statistics(df)
    df = treat_missing_data(df, [])

    # df = feature_generation(df, excluded_columns=["MEDV"])

    df = eda.normalize_data_sklearn(df, [])

    if exploratory_analysis:
        descriptive_statistics(df)
        # eda.analyze_class_skewness(df, []) # no categorical feature
        eda.analyze_distribution(df, layout=(4, 4))
        # outlier detection
        eda.analyze_correlation(df)

    # 3. Prepare Data
    # a) Data Cleaning
    # b) Feature Selection
    # c) Data Transforms

    X = df.drop(["MEDV"], axis=1)
    y = df["MEDV"]

    nfeatures = math.ceil(X.shape[1] / 3)

    if feature_selection:
        fselect.univariable_selection(X, y, nfeatures, classification_tasks=classification_tasks)
        fselect.RFE(X, y, nfeatures, classification_tasks=classification_tasks)
        fselect.tree_classifier(X, y, nfeatures, classification_tasks=classification_tasks)
        freduct.PCA(X, y, explained_variance=0.85)
        freduct.pca_show_variance_rate(X)

    # 4. Evaluate Algorithms
    # a) Split-out validation dataset
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,
                                                                    test_size=validation_size, random_state=seed)

    X = fselect.exhaustive_feature_selection(X_train, Y_train, num_features=[2, 10],
                                             classification_tasks=classification_tasks)

    scoring = 'mean_absolute_error'

    # train model
    kfold = skmodsel.KFold(n_splits=10, random_state=7)

    models = [["LReg", skmod.LinearRegression()],
              ["RIDGE", skmod.Ridge()],
              ["LASSO", skmod.Lasso()],
              ["ENET", skmod.ElasticNet()],
              # ["LDA", skdals.LinearDiscriminantAnalysis()],
              ["kMeans", skl.neighbors.KNeighborsRegressor()],
              ["DTREE", skl.tree.DecisionTreeRegressor()],
              ["SVM-L", sksvm.LinearSVR()],
              ["SVR-L", sksvm.SVR(kernel='linear')],
              ["SVR-RBF", sksvm.SVR(kernel='rbf')]
              ]

    # scoring = "accuracy"
    print("============= SIN Reducción ======================")
    summary = evalc.cross_val_summary_simple(models, X, y, scoring=scoring, options={"title": "NO RED"})
    print("===========================================")
    print("============= CON PCA ======================")
    X, scores = freduct.PCA(X, y, explained_variance=0.85)
    summary = evalc.cross_val_summary_simple(models, X, y, scoring=scoring, options={"title": "PCA"})
    print("===========================================")
    if classification_tasks:
        print("============= CON LDA ======================")
        # X, scores = freduct.LDA(X, y, explained_variance=0.85)
        # summary = evalc.cross_val_summary_simple(models, X, y, scoring=scoring, options={"title": "LDA"})
        print("===========================================")

    if model_evaluation and classification_tasks:
        print("============== MODEL EVALUATION ===================")

        y_enc = np.zeros(y.shape)
        y_enc[y == "Female"] = 1

        summary = evalc.cross_val_summary(models, X, y_enc, metrics=["accuracy", "roc_auc"])

        y_true, y_predicted = evalc.train_model(models[0][1], X, y)
        cmtx = skmet.confusion_matrix(y_true, y_predicted)
        class_names = np.unique(y)
        evalc.plot_confusion_matrix(cmtx, classes=class_names, normalize=True)

        report = skmet.classification_report(y_true, y_predicted)
        print(report)

    # b) Test options and evaluation metric
    # c) Spot Check Algorithms
    # d) Compare Algorithms

    # 5. Improve Accuracy
    # a) Algorithm Tuning
    # b) Ensembles

    # 6. Finalize Model
    # a) Predictions on validation dataset
    # b) Create standalone model on entire training dataset
    # c) Save model for later use

    plt.show()
