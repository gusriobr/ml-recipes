import numpy as np
import pandas as pd
import scipy.stats as sst
import spm1d
from matplotlib import pyplot as plt


def normality_test(df, excluded_columns=[], alpha=0.05):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    df_selected = df[selected_cols]
    test_results = {}
    for col in df_selected:
        values = df_selected[col]
        res = sst.mstats.normaltest(values)
        test_results[col] = {"statistic": res.statistic, "pvalue": res.pvalue, "alpha": alpha,
                             "rejectH0": (res.pvalue < alpha)}
    return pd.DataFrame(test_results).transpose()


def homocedas_barlett(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    data = df[selected_cols].values.T.astype(np.float).tolist()
    ret = sst.bartlett(*data)
    return ret

def homocedas_levene(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    data = df[selected_cols].values.T.astype(np.float).tolist()
    ret = sst.levene(*data)
    return ret

def homocedas_kruskal(df, excluded_columns=[]):
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    data = df[selected_cols].values.T.astype(np.float).tolist()
    ret = sst.kruskal(*data)
    return ret



def anova_plot_group_means(df, categorical_variable):
    # plot profile of group means
    cat_values = df[categorical_variable]
    means = df.groupby(by=categorical_variable).mean()

    plt.figure()
    legend_values = means.columns.values.tolist()
    plt.plot(means)
    plt.legend(legend_values, loc=1)
    plt.title("Perfiles de medias por factor")


def __segmentation_by_category(df, category):
    cat_values = np.unique(df[category])
    df_list = []
    for v in cat_values:
        dt_segment = df.loc[df[category] == v]
        dt_segment = dt_segment.drop([category], axis=1)
        df_list.append(dt_segment.values)

    return df_list


def anova_test(df, categorical_variable):
    import scipy.stats as stats
    segments = __segmentation_by_category(df, categorical_variable)

    # Perform the ANOVA
    test_result = stats.f_oneway(*segments)
    # http://hamelg.blogspot.com.es/2015/11/python-for-data-analysis-part-16_23.html
    # http://www.spm1d.org/doc/Stats1D/multivariate.html#one-way-manova
    return test_result


def manova_test(df, categorical_variable, excluded_columns=[], alpha=0.05):
    y = df[categorical_variable]
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    X = df[selected_cols]
    X = X.drop([categorical_variable], axis=1)

    model = spm1d.stats.manova1(X, y)
    inf = model.inference(alpha)
    return inf
