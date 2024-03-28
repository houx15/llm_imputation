import pandas as pd
import numpy as np

import os
import json

from sklearn import metrics

from typing import List


inapplicable_sig = ".i"
inapplicable_sig2 = "iap"
id_columns = ["id", "wtssnrps", "wtssps", "wtssnrps_next", "wtssps_next"]

text_stop_words = [
    "iap",
    "skipped on web",
]

same_features = {"income16": ["income", "coninc", "realinc"], "educ": ["degree"]}


class DatasetPreparation(object):
    def __init__(self) -> None:
        self.init_var_label_dict()
        self.to_predict_features = ["income16", "educ", "letin1a", "finrela", "marhomo"]

    def init_var_label_dict(self):
        var_label_df = pd.read_csv("dataset/label.csv")
        self.var_label_dict = {}
        for index, row in var_label_df.iterrows():
            self.var_label_dict[row["name"]] = row["simplab"]

        del var_label_df

    def init_dataset(self, target_features: List[str], skip_predicting_labels: bool):
        dataset = self.numeric_data_preparation()

        for target_feature in target_features:
            dataset = dataset[dataset[target_feature] != inapplicable_sig]
            dataset.dropna(axis=0, subset=[target_feature], inplace=True)
        dataset.replace(inapplicable_sig, np.nan, inplace=True)
        dataset.replace(np.nan, -9, inplace=True)
        self.id_remained = dataset["id"].values
        dataset = dataset.apply(pd.to_numeric, errors="coerce")
        # dataset.drop(id_columns, axis=1, inplace=True)
        # TODO id_columns不应该在这个时候就被删除

        self.dataset = dataset
        self.skip_predicting_labels = skip_predicting_labels
        return self.dataset

    def numeric_data_preparation(self):
        data = pd.read_csv("dataset/num_2022.csv")
        i_num = data.apply(lambda col: col[col == ".i"].count()).sum()
        n_num = data.apply(lambda col: col[col == ".n"].count()).sum()
        d_num = data.apply(lambda col: col[col == ".d"].count()).sum()

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

    def mutual_information(self, applicable, target_feature):
        # 选择互信息最高的n个feature
        features = applicable.drop(target_feature, axis=1)

        stop_feat = same_features.get(target_feature, [])
        # TODO Delete to predict features
        all_stop_feat = stop_feat + id_columns
        if self.skip_predicting_labels:
            self.to_predict_features.remove(target_feature)
            all_stop_feat += self.to_predict_features
        features = features.drop(all_stop_feat, axis=1)

        target = applicable[target_feature]

        muinfo = {"feature": [], "info": []}
        for c in features.columns:
            info = metrics.mutual_info_score(features[c], target)
            muinfo["feature"].append(c)
            muinfo["info"].append(info)

        output_df = pd.DataFrame(muinfo)
        output_df.sort_values(by="info", ascending=False, inplace=True)
        return output_df

    def prompt_compiler(self, row, target_feature):
        prompt = "One's data in 2022 GSS data: "
        for var, value in row.iteritems():
            if var == "id":
                continue
            if var == target_feature:
                continue
            if value == "iap":
                continue
            var_label = self.var_label_dict.get(var)
            if var_label is None:
                continue

            prompt += f"{var_label} is {value}. "
        return prompt

    def get_text_dataset(self, top_keys, target_feature, id_remained):
        text_df = pd.read_csv("dataset/text_2022.csv")

        text_df.drop(text_df[~text_df["id"].isin(id_remained)].index, inplace=True)
        text_label = text_df[target_feature]

        remained_keys = ["id"] + top_keys + [target_feature]
        text_df = text_df[remained_keys]

        target_label = self.var_label_dict[target_feature]
        suffix = f"Please imputate his {target_label}."

        text_df["text"] = text_df.apply(
            lambda x: self.prompt_compiler(x, target_feature), axis=1
        )
        text_df["text"] = text_df["text"] + suffix

        text_df.rename(columns={target_feature: "labels"}, inplace=True)

        return text_df

    def to_json(self, row):
        number_array = row.to_numpy()  # 获取一行的值，并转换为 NumPy 数组
        json_str = json.dumps(number_array.tolist())  # 转换为 JSON 字符串
        return json_str

    def get_feature_full_dataset(
        self, target_feature, datapath=None, feature_num=50, strategy="numeric"
    ):
        mutual_info = self.mutual_information(self.dataset, target_feature)
        top_keys = mutual_info[:feature_num]["feature"].to_list()

        remained_keys = ["id"] + top_keys + [target_feature]
        output_dataset = self.dataset[remained_keys]
        # label = self.dataset[target_feature]

        text_dataset = self.get_text_dataset(top_keys, target_feature, self.id_remained)
        text_dataset = text_dataset[["text", "labels", "id"]]

        output_dataset["json_text"] = output_dataset.apply(self.to_json, axis=1)
        output_dataset = output_dataset[["json_text", "id", target_feature]]

        all_dataset = pd.merge(left=output_dataset, right=text_dataset, on=["id"])
        # all_dataset.drop(
        #     all_dataset[all_dataset[target_feature] == -9].index, inplace=True
        # )
        # all_dataset.drop(
        #     all_dataset[all_dataset[target_feature] == None].index, inplace=True
        # )
        # all_dataset.dropna(axis=0, subset=[target_feature], inplace=True)

        all_dataset.rename(columns={target_feature: "num_label"}, inplace=True)

        output_dataset = all_dataset[["json_text", "num_label"]]
        text_dataset = all_dataset[["text", "labels", "num_label"]]
        text_dataset.dropna(axis=0, subset=["text"], inplace=True)

        if datapath is None:
            datapath = f"dataset/{target_feature}"
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        text_dataset.to_csv(f"{datapath}/text-dataset.csv", index=False)
        output_dataset.to_csv(f"{datapath}/num-dataset.csv", index=False)
        return output_dataset, text_dataset
