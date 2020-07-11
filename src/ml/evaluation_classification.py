import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import train_test_split, KFold


def evaluate_model(models, X, y, splits_kfold=10, scoring=None, options={}):
    import sklearn.metrics as skmet

    title = 'Model comparative ';
    if "title" in options:
        title += " - " + options["title"];

    map_results = {}
    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        results = []
        y_predict = model.fit(X_train, y_train).predict(X_test)
        if scoring is None:
            scoring = skmet.mean_squared_error

        scores = scoring(y_true=y_test, y_pred=y_predict)
        results.append(scores)

        map_results[name] = np.array(results)

    df_results = get_cv_results_df(map_results, scoring.__name__ )
    plot_cv_results(df_results, title=title)
    return df_results


def cross_val_summary_simple(models, X, y, scoring="accuracy", splits_kfold=5, options={}):
    kfold = KFold(n_splits=splits_kfold, random_state=7)

    map_results = {}
    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        map_results[name] = cv_results

    df_results = get_cv_results_df(map_results, scoring)

    title = 'Model comparative ';
    if "title" in options:
        title += " - " + options["title"];

    plot_cv_results(df_results, title)

    return df_results


def get_cv_results_df(map_results, scoring_func_name):
    """
    Creates a dataframe with one observation per model test result
    """

    df_results = pd.DataFrame(data=[], columns=['model', 'scoring', 'value'])
    for model_name, cv_results in map_results.items():
        df_row = pd.DataFrame(cv_results.T, columns=['value'])
        df_row['model'] = model_name
        df_row['scoring'] = scoring_func_name
        df_results = df_results.append(df_row)

    return df_results


def plot_cv_results(df_results, title="Model comparative", print_summary=True, order_by="score"):
    """
    Creates boxplot chart to show cross validation results
    :param df_results:
    :param title:
    :param order_by: score | model
    :return:
    """
    # calculate mean and std
    scoring_func_name = df_results["scoring"].unique()

    df_summary = df_results.groupby(["model"]).agg(["mean", "std"]).reset_index()
    df_summary.columns = ["model", "mean_val", "std_val"]

    model_names = df_summary["model"].tolist()
    if order_by != "score":
        model_names.sort()

    if order_by == "score":
        df_summary = df_summary.sort_values(by=["mean_val"], ascending=[False])
        model_names = df_summary["model"].tolist()

    plt.figure()
    ax = sns.boxplot(x="model", y="value", data=df_results, palette="Set3", order=model_names)
    # ax = sns.swarmplot(x="model", y="value", data=df_results, color=".25")
    ax = sns.stripplot(x="model", y="mean_val", data=df_summary, color='red', order=model_names)
    ax.set_title(title)

    # for index, row in mean_values.iterrows():
    #     ax.text(row.name, row.value + 0.02, "%s(%s)" % (round(row.value, 2),
    #                                                     round(std_values.loc[row.model].value, 2)),
    #             color='black', ha="center")


    pos = 0
    for index, row in df_summary.iterrows():
        ax.text(pos, row.mean_val + 0.06, "m:%s" % (round(row.mean_val, 2)),
                color='black', ha="center")
        ax.text(pos, row.mean_val + 0.02, "s:%s" % (round(row.std_val, 2)),
                color='black', ha="center")
        if print_summary:
            msg = "%s: %s %.4f (%.4f)" % (scoring_func_name, row.model, row.mean_val, row.std_val)
            print(msg)
        pos += 1
    return


def cross_val_summary(model_map, X, y, metrics=None):  # =['accuracy', 'neg_log_loss', 'roc_auc']):
    """
    Creates dataframe to store metric values realated to each model
    :param model_map:
    :param X:
    :param y:
    :param metrics:
    :return:
    """
    kfold = StratifiedKFold(n_splits=10, random_state=7)

    df = None
    columns = None
    for model in model_map:
        results = cross_validate(model[1], X, y, cv=kfold, scoring=metrics)
        if not columns:
            # initilize dataframe
            columns = list(results.keys())
            cols = [x + "_mean" for x in columns]
            cols.extend([x + "_std" for x in columns])
            cols = ["Model"] + cols
            df = pd.DataFrame(columns=cols)

        row = {}
        row["Model"] = model[0]
        row.update({k + "_mean": round(v.mean(), 3) for k, v in results.items()})
        row.update({k + "_std": round(v.std(), 3) for k, v in results.items()})

        df = df.append(row, ignore_index=True)

    df.set_index((['Model']))
    return df


def train_model(model, X, y, test_size=0.33, seed=7):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return Y_test, predicted


def plot_confusion_matrix(cnf_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          output_file = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output_file is not None:
        plt.savefig(output_file)
