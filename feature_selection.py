from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

from sklearn import metrics


inapplicable_sig = ".i"
id_columns = ["id", "wtssnrps", "wtssps", "wtssnrps_next", "wtssps_next"]


def data_preparation():
    data = pd.read_csv("dataset/num_2022.csv")
    i_num = data.apply(lambda col: col[col == '.i'].count()).sum()
    n_num = data.apply(lambda col: col[col == '.n'].count()).sum()
    d_num = data.apply(lambda col: col[col == '.d'].count()).sum()

    print(f"dataset contains {i_num} .i, {d_num} .d, and {n_num} .n")

    # data.replace(".i", np.nan, inplace=True)
    data.replace(".n", np.nan, inplace=True)
    data.replace(".d", np.nan, inplace=True)
    data.replace(".s", np.nan, inplace=True)
    data.replace(".r", np.nan, inplace=True)
    data.replace(".x", np.nan, inplace=True)
    data.replace(".y", np.nan, inplace=True)
    data.replace(".z", np.nan, inplace=True)
    data.replace(".u", np.nan, inplace=True)
    data.replace(".m", np.nan, inplace=True)
    data.replace(".b", np.nan, inplace=True)
    data.replace(".p", np.nan, inplace=True)
    data.replace(".f", np.nan, inplace=True)

    return data

def importance(data, target_feature):
    applicable = data[data[target_feature] != inapplicable_sig]
    applicable.replace(inapplicable_sig, np.nan, inplace=True)
    applicable.replace(np.nan, -9, inplace=True)
    applicable = applicable.apply(pd.to_numeric, errors='coerce')
    applicable = applicable.drop(id_columns, axis=1)

    features = applicable.drop(target_feature, axis=1)
    target = applicable[target_feature]
    # print(max(target), min(target), target_feature)

    rf = RandomForestRegressor()
    rf.fit(features, target)

    feature_importance = rf.feature_importances_
    feature_importance_dict = dict(zip(features.columns, feature_importance))
    # print(feature_importance_dict)
    output_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
    output_df.sort_values(by="Importance", ascending=False, inplace=True)
    output_df.to_excel(f"temp_data/feature_importance_{target_feature}.xlsx", index=False)

def mutual_information(data, target_feature):
    applicable = data[data[target_feature] != inapplicable_sig]
    applicable.replace(inapplicable_sig, np.nan, inplace=True)
    applicable.replace(np.nan, -9, inplace=True)
    applicable = applicable.apply(pd.to_numeric, errors='coerce')
    applicable = applicable.drop(id_columns, axis=1)

    features = applicable.drop(target_feature, axis=1)
    target = applicable[target_feature]

    muinfo = {"feature": [], "info": []}
    for c in features.columns:
        info = metrics.mutual_info_score(features[c], target)
        muinfo["feature"].append(c)
        muinfo["info"].append(info)
    
    output_df = pd.DataFrame(muinfo)
    output_df.sort_values(by="info", ascending=False, inplace=True)
    output_df.to_excel(f"temp_data/mutual_information_{target_feature}.xlsx", index=False)


def data_show(data):
    filtered_columns = data.columns[(data == '.i').sum() <= 1500]
    nan_counts_per_column = data[filtered_columns].isna().sum()
    i_counts_per_column = (data[filtered_columns] == '.i').sum()
    results = pd.concat([nan_counts_per_column, i_counts_per_column], axis=1)
    results.columns = ['NaN Counts', '.i Counts']
    sorted_nan_counts = results.sort_values(by='NaN Counts', ascending=False)
    sorted_nan_counts.to_excel("temp_data/no_answer_2.xlsx")

# model = RandomForestClassifier()
# model.fit()
data = data_preparation()
# inapplicable的不应该用来预测

objective_features = ["income16", "raclive", "age", "pres20", "mawrkgrw", "wordsum", "educ"]
opinion_features = ["cappun", "polviews", "letin1a", "homosex", "premarsx", "finrela", "gunlaw", "class", "marhomo", "partyid"]

for objfea in objective_features:
    importance(data, objfea)
    # mutual_information(data, objfea)

for opfea in opinion_features:
    importance(data, opfea)
    # mutual_information(data, opfea)