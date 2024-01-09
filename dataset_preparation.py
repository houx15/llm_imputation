import pandas as pd
import numpy as np

import os

from sklearn import metrics


inapplicable_sig = ".i"
id_columns = ["id", "wtssnrps", "wtssps", "wtssnrps_next", "wtssps_next"]

text_stop_words = ["iap", "skipped on web", ]


def numeric_data_preparation():
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


def mutual_information(applicable, target_feature):
    # 选择互信息最高的n个feature

    features = applicable.drop(target_feature, axis=1)
    target = applicable[target_feature]

    muinfo = {"feature": [], "info": []}
    for c in features.columns:
        info = metrics.mutual_info_score(features[c], target)
        muinfo["feature"].append(c)
        muinfo["info"].append(info)
    
    output_df = pd.DataFrame(muinfo)
    output_df.sort_values(by="info", ascending=False, inplace=True)
    return output_df


def get_text_dataset(top_keys, target_feature):
    var_label_df = pd.read_csv("dataset/label_2022.csv")
    var_label_dict = {}
    for index, row in var_label_df.iterrows():
        var_label_dict[row["name"]] = row["varlab"]
    
    del var_label_df

    text_df = pd.read_csv("dataset/text_2022.csv")
    text_label = text_df[target_feature]
    text_df = text_df[top_keys]
    
    text_data_dict = {"text": [], "label": []}

    target_label = var_label_dict[target_feature]
    suffix = f"Please imputate what the {target_label} of this individual is."
    for index, row in text_df.iterrows:
        label = text_label.loc[index]

        prompt = "Here is the data for an individual in a survey dataset."
        for var, value in row.iteritems():
            if value == "iap":
                continue
            var_label = var_label_dict.get(var)
            if var_label is None:
                continue

            prompt += f"The {var_label} is {value}. "
        
        prompt += suffix

        text_data_dict["text"].append(prompt)
        text_data_dict["label"].append(label)
    
    text_data_df = pd.DataFrame(text_data_dict)
    return text_data_df




def get_full_dataset(target_feature, datapath=None, feature_num=50, strategy="numeric"):
    dataset = numeric_data_preparation()

    dataset = dataset[dataset[target_feature] != inapplicable_sig]
    dataset.replace(inapplicable_sig, np.nan, inplace=True)
    dataset.replace(np.nan, -9, inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset = dataset.drop(id_columns, axis=1)

    mutual_info = mutual_information(dataset, target_feature)
    top_keys = mutual_info[:feature_num]["feature"]


    output_dataset = dataset[top_keys]
    label = dataset[target_feature]

    text_dataset = get_text_dataset(top_keys, target_feature)

    output_dataset = pd.concat([output_dataset, label], axis=1)
    text_dataset = pd.concat([text_dataset, label], axis=1)

    output_dataset.rename(columns={target_feature: "num_label"}, inplace=True)
    text_dataset.rename(columns={target_feature: "num_label"}, inplace=True)

    if datapath is None:
        datapath = f"dataset/{target_feature}"
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    text_dataset.to_csv(f"{datapath}/text-dataset.csv")
    output_dataset.to_csv(f"{datapath}/num-dataset.csv")
    return output_dataset, text_dataset
