from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import recipes.exploratory_analysis as eda


def config_printing():
    np.set_printoptions(linewidth=500)
    np.set_printoptions(precision=4)
    pd.set_option('display.width', 500)
    pd.set_option('precision', 4)


def read_data():
    data = pd.read_csv("brain_size.csv", sep=';', na_values=".")
    data = data.drop(['Unnamed: 0'], axis=1)
    return data

def treat_missing_data(df, categorical_cols):
    # 1 - Mark wrong values as NAN example
    # df[[1, 2, 3, 4, 5]] = df[[1, 2, 3, 4, 5]].replace(0, numpy.NaN)
    # 2 - Drop missing values
    # 3 - Imputer
    # replace missing data using column average value
    df.fillna(df.mean(), inplace=True)
    return df

print(__doc__)

# Data analysis
df = read_data()
# print_column_datatypes(df)
# descriptive_statistics(df)
df = treat_missing_data(df, ["Gender"])
df = eda.normalize_data_sklearn(df, ["Gender"])
y = df["Gender"]
X = df.drop(["Gender"], axis=1)

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [2, 4, 6]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()