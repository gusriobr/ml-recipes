import math

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing as skpre
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy import stats


def print_column_datatypes(df):
    print("\n========== Feature datatypes ============")
    print(df.dtypes)


def descriptive_statistics(df):
    print("\n========== Descriptive Statistics ============")
    # Things to look for:
    # Data that makes no sense: imposible min/max values: On some columns, a value of zero does not make sense,
    print(df.describe())


def analyze_class_skewness(df, features):
    print("\n========== Classes Skewness ============")
    for f in features:
        print(df[f].value_counts(normalize=True))


def analyze_correlation(df, excluded_columns=[], corr_method="pearson"):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]

    # Pearson's coeficient: asumes Normal distribution
    # Asummes linear dependency
    # Not robust in presence of outliers
    names = list(df[selected_cols].columns)
    title = "Data correlations method " + corr_method
    print("\n>>>> " + title)
    correlations = df[selected_cols].corr(method=corr_method)
    plot_correlations(title, correlations)

    return correlations


def sort_correlations(cormatrix, numtoreport=10):
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cormatrix.head(numtoreport)


def plot_pairplot(df, categorical_column=None, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]

    # __plot_correlations(names, correlations)
    f, ax = plt.subplots(figsize=(40, 40))
    plt.title("Data pairplot")
    sns.set(style="ticks")
    if categorical_column:
        sns.pairplot(df[selected_cols], hue=categorical_column)
    else:
        sns.pairplot(df[selected_cols])


def plot_parallel_coordinates(df, categorical_column, excluded_columns=[]):
    from pandas.plotting import parallel_coordinates

    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    y = df[categorical_column]
    df_norm = df[selected_cols].copy()
    df_norm = df_norm.drop(categorical_column, axis=1)

    df_norm = normalize_data_sklearn(df_norm)

    # Concat classes with the normalized data
    df_norm = pd.concat([df_norm, y], axis=1)

    # Perform parallel coordinate plot
    fig = plt.figure()
    parallel_coordinates(df_norm, categorical_column)


def plot_profile(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    fig = plt.figure()
    ax = df[selected_cols].plot()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


def plot_correlations(title, correlations, diagonal=True):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlations, dtype=np.bool)
    if diagonal:
        mask[np.triu_indices_from(mask)] = True
    else:
        mask[:] = True

    fig = plt.figure()
    f, ax = plt.subplots(figsize=(40, 40))
    plt.title(title)
    # plt.figure(figsize=(15, 15))
    sns.heatmap(correlations, mask=np.zeros_like(correlations, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                annot=True, square=True, ax=ax)


def __plot_correlations2(names, correlations):
    # print correlation matrix
    print(correlations)
    # draw matrix correlation graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)


def plot_correlations_heatmap(correlations):
    # correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    return


def hinton(df, categorical_column=None, excluded_columns=[], max_weight=None, ax=None):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    matrix = df[selected_cols].values
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def __hinton(df, categorical_column=None, excluded_columns=[], max_weight=None, ax=None):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    matrix = df[selected_cols]
    matrix = df[selected_cols].values

    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()


def profile_plot(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]

    ax = df[selected_cols].plot()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


def normalize_data(df, excluded_columns=[]):
    # remove categorical data
    df_aux = df
    cols = None
    if excluded_columns:
        cols = df_aux[excluded_columns]
        df_aux = df_aux.drop(excluded_columns, axis=1)
    df_norm = (df_aux - df_aux.min()) / (df_aux.max() - df_aux.min())

    return pd.concat([df_norm, cols], axis=1)


def normalize_data_sklearn(df, excluded_columns=[]):
    """
    :param df:
    :param excluded_columns: categorical/ordinal columns, and descriptive columns to dismiss for normalization
    :return:
    """

    df_excluded = None
    if excluded_columns:
        df_excluded = df[excluded_columns]

    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    values = skpre.MinMaxScaler().fit_transform(df[selected_cols])
    new_df = pd.DataFrame(values, columns=df[selected_cols].columns)
    if df_excluded is not None:
        new_df = pd.concat([new_df, df_excluded], axis=1)
    return new_df


def standardize_data(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    df[selected_cols] = (df[selected_cols] - df[selected_cols].mean()) / df[selected_cols].std()
    return df


def standardize_data_sklearn(df, excluded_columns=[]):
    df_excluded = None
    if excluded_columns:
        df_excluded = df[excluded_columns]

    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    values = skpre.StandardScaler().fit_transform(df[selected_cols])
    new_df = pd.DataFrame(values, columns=df[selected_cols].columns)
    if df_excluded is not None:
        new_df = pd.concat([new_df, df_excluded], axis=1)
    return new_df


def analyze_distribution(df, layout=(3, 3)):
    print("\n========== Data Distribution Analysis============")
    # plot histograms
    df.hist()
    # plot smooth distribution charts
    df.plot(kind='density', subplots=True, layout=layout, sharex=False)
    # draw whiskers charts
    df.plot(kind=' box ', subplots=True, layout=layout, sharex=False, sharey=False)
    # analyze skewness
    print("\n>>> Skewness")
    print(df.skew())


def plot_normal_probability(df, excluded_columns=[], normalize_data=False, n_columns=5):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    # selected_cols = [selected_cols[0]]

    data = df[selected_cols].copy()
    if normalize_data:
        data = normalize_data_sklearn(data)

    fit_data = []

    N = len(data.columns)
    cols = min(N, n_columns)

    rows = int(math.ceil(N / cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    for idx, col in enumerate(data):
        ax = fig.add_subplot(gs[idx])
        values = (data[col].values)
        res = stats.probplot(values, dist=stats.norm, sparams=(values.mean(), values.var()), plot=plt)
        ax.set_xlabel("{} - R: {}".format(col, round(res[1][2], 3)))
        ax.set_ylabel("")
        ax.set_title("")

        fit_data.append((col, res))

    fig.tight_layout()

    return fit_data
