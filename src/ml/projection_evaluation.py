import scipy.stats as stats
import pandas as pd
import os
import fritura.config as cfg
import os

import pandas as pd
import scipy.stats as stats

import fritura.config as cfg


def save_pca_analysis(projected_features, df_correlation):
    pca_file = os.path.join(cfg.get_resource_folder(), "PCA_correlacion_variables_new.xlsx")
    if os._exists(pca_file):
        os.remove(pca_file)
    writer = pd.ExcelWriter(pca_file, engine='openpyxl')
    df_correlation.to_excel(writer, sheet_name="pca_correlations")
    projected_features.to_excel(writer, sheet_name="projected_features")
    writer.save()


def pca_correlations(df_components, df, corr_func="pearson", excluded_columns=[], deeming_rate=0.5):
    # d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    #      ....: 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

    selected_cols = [x for x in df.columns.values if x not in excluded_columns]

    corr_matrix = {}
    relevant_relations = {}
    for component in df_components:
        values = []
        relevant_relations[component] = []
        for feature in df[selected_cols]:
            # calculate pairwise correlation value
            score = stats.pearsonr(df_components[component], df[feature])
            values.append(score[0])
            if score[0] >= deeming_rate:
                relevant_relations[component].append((feature, score[0]))

        corr_matrix[component] = pd.Series(values, index=df[selected_cols].columns)

    corr_df = pd.DataFrame(corr_matrix)

    return corr_df, relevant_relations

    return principal_componentes
