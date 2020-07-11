import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(df, excluded_columns=None, order=3):
    """
    Combines the dataframe features up to the given parameter order (default=3)
    :param df:
    :param excluded_columns: columns to exclude from the feature generation. Typically to
    avoid mix the target feature with other features
    :return:
    """
    poly = PolynomialFeatures(order, include_bias=False)
    selected_cols = [x for x in df.columns.values if x not in excluded_columns]
    X = poly.fit_transform(df[selected_cols])
    new_feature_names = poly.get_feature_names(selected_cols)
    new_feature_names.extend(excluded_columns)

    # add target columns to dataframe
    if excluded_columns:
        X = np.concatenate((X, df[excluded_columns].values), axis=1)
    return pd.DataFrame(X, columns=new_feature_names)
