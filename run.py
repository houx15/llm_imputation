import argparse
import json

from dataset_preparation import DatasetPreparation
from model import LlamaModel

import importlib
import fire
import os

from utils.utils import log
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Union

# objective_features = [
#     "income16",
#     "raclive",
#     "age",
#     "pres20",
#     "mawrkgrw",
#     "wordsum",
#     "educ",
# ]

objective_features = [
    "income16",
    "educ",
]

# opinion_features = [
#     "cappun",
#     "polviews",
#     "letin1a",
#     "homosex",
#     "premarsx",
#     "finrela",
#     "gunlaw",
#     "class",
#     "marhomo",
#     "partyid",
# ]

opinion_features = [
    "letin1a",
    "finrela",
    "marhomo",
]

model_dict = {
    "7b": "chainyo/alpaca-lora-7b",
    "13b": "yahma/llama-13b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}

peft_dict = {
    "7b": "PEFT/alpaca-lora-7b",
    "13b": "PEFT/alpaca-13b-lora",
    "llama-2-7b": None,
    "llama-2-13b-chat": None,
    "llama-2-13b": None,
}


def feature_prediction(
    features: List[str],  # TODO one feature or multi-features
    cache_results: bool = False,
    do_train: bool = False,
    do_eval: bool = False,
    peft: bool = True,
    use_pretrained_peft_weights: bool = False,
    model_type: str = "llama-2-13b",
    eval_model_path: str = None,
    parameter_search: bool = False,
    output_dir_base: str = None,
    output_dir: str = None,
    log_dir: str = None,
    resume_from_checkpoint: str = None,
    strategy: str = "numeric",  # "numeric"/"text"/"prompt"
    nfold: int = 10,
    skip_predicting_labels: bool = False,
):
    config_module = None
    config_module_name = f"train_para.configs-{strategy}"
    config_module = importlib.import_module(config_module_name)
    training_args = config_module.trainin_args

    if config_module is None:
        raise ValueError(
            "You need to specify config file or topic info to run this project"
        )

    if output_dir_base is None:
        output_dir_base = "/scratch/network/yh6580/"
        print(
            "Warning: you may encounter a permission error due to the wrong output base dir"
        )

    if model_dict.get(model_type) is None:
        raise NotImplementedError(f"Model type {model_type} is not implemented!")

    data_pre = DatasetPreparation()
    all_data = data_pre.init_dataset(
        target_features=features, skip_predicting_labels=skip_predicting_labels
    )

    for i in range(nfold):
        # run 1 fold TODO
        fold_size = len(all_data) // nfold
        start = i * fold_size
        end = (i + 1) * fold_size if i < nfold - 1 else len(all_data)

        test_indices = list(range(start, end))
        if i > 0:
            break
        all_true = []
        all_pred = []
        for feature in features:
            try:
                assert cache_results is True
                result_df = pd.read_csv(f"dataset/{feature}/result.csv")
                true = result_df["num_label"].values
                pred = result_df["prediction"].values
            except:
                if feature in objective_features:
                    print(f"Running objective feature {feature}")
                elif feature in opinion_features:
                    print(f"Running opinion feature {feature}")
                else:
                    raise NotImplementedError(f"Feature {feature} is unexpected")
                output_dir = (
                    f"{output_dir_base}output/{strategy}/{model_type}/{feature}"
                )
                log_dir = f"{output_dir_base}logs/{strategy}/{model_type}/{feature}"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_file = f"{log_dir}/result.txt"

                with open("configs/task_description.json") as fp:
                    task_description = json.load(fp)
                    task_description = task_description[feature]

                if task_description["task_type"] == "class":
                    print(f"feature {feature} has a task type of class!")
                    return

                log(log_file, f">>>>>>>>>>>>>>>>>running feature {feature} fold {i+1}")
                datapath = f"dataset/{feature}"
                full_dataset, text_dataset = data_pre.get_feature_full_dataset(
                    target_feature=feature, datapath=datapath, feature_num=120
                )
                print(full_dataset.isna().sum(), text_dataset.isna().sum())
                # full_dataset, text_dataset = get_full_dataset(feature, datapath=datapath)
                dataset = full_dataset if strategy == "numeric" else text_dataset

                test_set = dataset.iloc[test_indices]
                train_set = dataset.drop(test_set.index)

                test_set.to_csv(f"{datapath}/test.csv", index=False)
                train_set.to_csv(f"{datapath}/train.csv", index=False)

                # hp_space_optuna = None
                # if hasattr(config_module, "hp_space_optuna"):
                #     hp_space_optuna = config_module.hp_space_optuna
                peft_weights = peft_dict.get(model_type)
                if do_eval:
                    if eval_model_path is None:
                        eval_model_path = output_dir
                    use_pretrained_peft_weights = True
                    peft_weights = eval_model_path

                model = LlamaModel(
                    feature=feature,
                    task_type=task_description["task_type"],
                    num_labels=task_description["num_labels"],
                    data_path=datapath,
                    strategy=strategy,
                    base_model=model_dict[model_type],
                    model_type=model_type,
                    param_dict=training_args.get(
                        f"{feature}-{model_type}", training_args.get("default", {})
                    ),
                    peft=peft,
                    peft_weights=peft_weights if use_pretrained_peft_weights else None,
                    output_dir=output_dir,
                    log_dir=log_dir,
                    resume_from_checkpoint=resume_from_checkpoint,
                    hp_space_optuna=None,
                )

                model.train()
                true, pred = model.eval()  # np array
            all_true.append(true)
            all_pred.append(pred)
        all_true = np.vstack(all_true)
        all_pred = np.vstack(all_pred)
        true_cov = np.cov(all_true)
        pred_cov = np.cov(all_pred)
        print(">>>>>>>>>>>>>>>true value covariance matrix")
        print(true_cov)
        print(">>>>>>>>>>>>>>>predict value covariance matrix")
        print(pred_cov)

        diff_cov = true_cov - pred_cov
        fig, (ax, ax2) = plt.subplots(nrows=2)
        sns.heatmap(true_cov, annot=True, ax=ax, vmin=-3, vmax=3, cmap="coolwarm")
        sns.heatmap(pred_cov, annot=True, ax=ax2, vmin=-3, vmax=3, cmap="coolwarm")

        plt.setp(
            (ax, ax2),
            xticklabels=["income16", "educ", "letin1a", "finrela", "marhomo"],
            yticklabels=["income16", "educ", "letin1a", "finrela", "marhomo"],
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        # ax.title("Covariance Difference (2024-01-22)")
        plt.savefig(
            "logs/cov_diff.pdf",
        )

        frobenius_norm = np.linalg.norm(true_cov - pred_cov, "fro")
        print(f"Frobenius norm: {frobenius_norm}")
    return all_true, all_pred


def run_all_features(
    mode: str = "multi",  # multi/single prediction
    cache: bool = False,
    do_train: bool = False,
    do_eval: bool = False,
    peft: bool = True,
    use_pretrained_peft_weights: bool = False,
    model_type: str = "llama-2-13b",
    eval_model_path: str = None,
    parameter_search: bool = False,
    output_dir_base: str = None,
    output_dir: str = None,
    log_dir: str = None,
    resume_from_checkpoint: str = None,
    strategy: str = "numeric",  # "numeric"/"text"/"prompt"
    nfold: int = 5,
    feature: str = None,
    skip_predicting_labels: bool = False,
):
    all_features = objective_features + opinion_features
    if feature is not None:
        assert feature in all_features
        all_features = [feature]

    if mode == "single":
        for f in all_features:
            true, pred = feature_prediction(
                features=[f],
                cache_results=cache,
                do_train=do_train,
                do_eval=do_eval,
                peft=peft,
                use_pretrained_peft_weights=use_pretrained_peft_weights,
                model_type=model_type,
                eval_model_path=eval_model_path,
                parameter_search=parameter_search,
                output_dir_base=output_dir_base,
                output_dir=output_dir,
                log_dir=log_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                strategy=strategy,
                nfold=nfold,
            )
        print(">>>>>>>>>>>>>>>>>>finished!")
    elif mode == "multi":
        true, pred = feature_prediction(
            features=all_features,
            cache_results=cache,
            do_train=do_train,
            do_eval=do_eval,
            peft=peft,
            use_pretrained_peft_weights=use_pretrained_peft_weights,
            model_type=model_type,
            eval_model_path=eval_model_path,
            parameter_search=parameter_search,
            output_dir_base=output_dir_base,
            output_dir=output_dir,
            log_dir=log_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            strategy=strategy,
            nfold=nfold,
            skip_predicting_labels=skip_predicting_labels,
        )
        print(">>>>>>>>>>>>>>>>>>finished!")


if __name__ == "__main__":
    fire.Fire(run_all_features)
