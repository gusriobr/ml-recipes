import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def pca_show_variance_rate(X):
    from sklearn.decomposition import PCA
    # feature extraction

    pca = PCA()
    fit = pca.fit(X)

    var_ratio = pca.explained_variance_ratio_
    var_ratio_accum = [ round(var_ratio[0:x].sum(),2) for x in range(1,len(var_ratio)+1)]

    plt.figure()
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(var_ratio, linewidth=2)
    plt.plot(var_ratio_accum, linewidth=2)
    plt.axis('tight')
    plt.grid(True)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')


def PCA(X, y, n_components=None, explained_variance = None):
    from sklearn.decomposition import PCA

    if n_components:
        pca = PCA(n_components=n_components)
        fit = pca.fit(X)
        num_selected = n_components
    else:
        num_selected = 0
        for n_comp in range(1, len(X.columns)):
            num_selected = n_comp
            # iterate until desired variance is reached
            pca = PCA(n_components=n_comp)
            fit = pca.fit(X)
            if(fit.explained_variance_ratio_.sum() > explained_variance):
                break

    features = pca.transform(X)
    col_names = [ "col_"+str(i) for i in range(0,num_selected)]
    features = pd.DataFrame(features, columns=col_names)
    features_score = fit.explained_variance_ratio_

    print ("============== Feature scores - PCA ===========")
    print("Explained Variance: {} = {}".format(features_score, features_score.sum()))
    # print("Selected Features: {}".format(features))

    return features, features_score


def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        print(summary)
    return summary


def LDA(X, y, n_components=3, explained_variance = None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if n_components:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        fit = lda.fit(X,y)
        num_selected = n_components
    else:
        num_selected = 0
        for n_comp in range(1, len(X.columns)):
            num_selected = n_comp
            # iterate until desired variance is reached
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            fit = lda.fit(X,y)
            if(fit.explained_variance_ratio_.sum() > explained_variance):
                break

    features = lda.transform(X)
    if features.shape[1] < num_selected:
        num_selected = features.shape[1]

    col_names = [ "col_"+str(i) for i in range(0,num_selected)]
    features = pd.DataFrame(features, columns=col_names)
    features_score = fit.explained_variance_ratio_

    print ("============== Feature scores - LDA ===========")
    print("Explained Variance: {}".format(fit.explained_variance_ratio_))
    # print("Selected Features: {}".format(features))

    return features, features_score