import numpy as np
import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


def univariable_selection(X, y, num_features=4, classification_tasks=True, score_func = None):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_regression

    if score_func is None:
        if classification_tasks:
            score_func = chi2
        else:
            score_func = f_regression

    test = SelectKBest(score_func=score_func, k=num_features)

    fit = test.fit(X, y)
    # summarize scores
    np.set_printoptions(precision=3)
    # get columns names

    col_names = X.columns
    # get the lower limit of selected features
    scores = fit.scores_
    limit = sorted(scores)[-num_features]

    # get selected features names
    col_idx = scores >= limit
    feature_names = col_names[col_idx]
    score_values = scores[col_idx]
    feature_scores = pd.DataFrame(score_values, index=feature_names, columns=["scores"])
    feature_scores = feature_scores.sort_values(by="scores", ascending=False)

    features = fit.transform(X)
    # summarize selected features
    print("============== Feature scores - Univariate - "+score_func.__name__+" ===========")
    print("Feature Ranking:\n {}".format(feature_scores))
    print("Selected Features: {}".format(features))

    # mutual_info_classif
    return features, feature_scores


def RFE(X, y, num_features=4, classification_tasks=True, model=None):
    """
    Implements feature selection using Recursive Feature Elimination
    :return:
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression, LinearRegression

    # column values
    col_names = X.columns

    if not model:
        if classification_tasks:
            model = LogisticRegression()
        else:
            model = LinearRegression()

    rfe = RFE(model, n_features_to_select=num_features)
    fit = rfe.fit(X, y)
    features = rfe.transform(X)

    feature_scores = pd.DataFrame(fit.ranking_, index=col_names, columns=["scores"])
    # best features has a #1 score value
    feature_scores = feature_scores.sort_values(by="scores", ascending=True)

    print("============== Feature scores - RFE ===========")
    print("Feature Ranking:\n {}".format(feature_scores))
    print("Selected Features: {}".format(features))

    return features, feature_scores


def tree_classifier(X, y, num_features=4, classification_tasks=True, model=None):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    # column values
    col_names = X.columns

    if not model:
        if classification_tasks:
            model = ExtraTreesClassifier()
        else:
            model = ExtraTreesRegressor()

    fit = model.fit(X, y)

    feature_scores = pd.DataFrame(model.feature_importances_, index=col_names, columns=["scores"])
    feature_scores = feature_scores.sort_values(by="scores", ascending=False)

    print("============== Feature scores - ExtraTree ===========")
    print("Feature Ranking:\n {}".format(feature_scores))

    return [], feature_scores


def lassoCV(X, y, num_features=4):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV()

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > num_features:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    feature1 = X_transform[:, 0]
    feature2 = X_transform[:, 1]


def exhaustive_feature_selection(x_train, y_train, model=None, num_features=[2, 5], classification_tasks=True,
                                 scoring=None):
    print("============== Exhaustive feature selection ===================")
    if not model:
        if classification_tasks:
            model = LogisticRegression(multi_class='multinomial',
                                       solver='lbfgs',
                                       random_state=123)
        else:
            model = Ridge()

    if not scoring:
        if classification_tasks:
            scoring = "accuracy"
        else:
            scoring = "neg_mean_absolute_error"

    efs = EFS(estimator=model,
               min_features=num_features[0],
               max_features=num_features[1],
               scoring=scoring,
               print_progress=False,
               clone_estimator=False,
               cv=10,
               n_jobs=2)

    X = efs.fit(x_train.values, y_train.values)
    print('Best accuracy score: %.2f' % efs.best_score_)
    col_list = []
    col_list.extend(efs.best_idx_)
    col_names = x_train.columns
    print('Best subset:', col_names[col_list].values)
    x_train = x_train.iloc[:,col_list]

    print("=================================")
    return x_train