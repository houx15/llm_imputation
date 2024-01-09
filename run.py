import argparse
import json

from dataset_preparation import get_full_dataset

objective_features = ["income16", "raclive", "age", "pres20", "mawrkgrw", "wordsum", "educ"]
opinion_features = ["cappun", "polviews", "letin1a", "homosex", "premarsx", "finrela", "gunlaw", "class", "marhomo", "partyid"]

model_dict = {
    "7b": "chainyo/alpaca-lora-7b",
    "13b": "yahma/llama-13b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}

def run(
        feature: str,
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
        strategy: str = "numeric", #"numeric"/"text"/"prompt"
        nfold: int = 10,
):
    if feature in objective_features:
        print(f"Running objective feature {feature}")
    elif feature in opinion_features:
        print(f"Running opinion feature {feature}")
    else:
        raise NotImplementedError(f"Feature {feature} is unexpected")
    
    if output_dir_base is None:
        output_dir_base = "/scratch/network/yh6580/"
        print(
            "Warning: you may encounter a permission error due to the wrong output base dir"
        )

    if model_dict.get(model_type) is None:
        raise NotImplementedError(
            f"Model type {model_type} is not implemented!"
        )

    if output_dir is None:
        output_dir = f"{output_dir_base}output/{strategy}/{model_type}/{feature}"
    if log_dir is None:
        log_dir = f"logs/{strategy}/{model_type}/{feature}"

    task_description = json.laods("configs/task_description.json")
    task_description = task_description[feature]
    
    # hp_space_optuna = None
    # if hasattr(config_module, "hp_space_optuna"):
    #     hp_space_optuna = config_module.hp_space_optuna
    datapath = f"dataset/{feature}"
    full_dataset, text_dataset = get_full_dataset(feature, datapath=datapath)
    
    if nfold is not None:
        fold_size = len(full_dataset) // nfold
        for i in range(nfold):
            start = i * fold_size
            end = (i + 1) * fold_size if i < nfold - 1 else len(full_dataset)

            test_indices = list(range(start, end))

            if strategy == "numeric":
                test_set = full_dataset.iloc[test_indices]
                train_set = full_dataset.drop(test_indices)
            else:
                test_set = text_dataset.iloc[test_indices]
                train_set = text_dataset.drop(test_indices)
            test_set.to_csv(f"{datapath}/test.csv")
            train_set.to_csv(f"{datapath}/train.csv")

            
