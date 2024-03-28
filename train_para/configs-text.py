trainin_args = {
    "default": {
        "batch_size": 32,
        "micro_batch_size": 32,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "cutoff_len": 900,
        "logging_steps": 10,
        "eval_steps": 10,
        "save_steps": 40,
    },
}


def hp_space_optuna(trial):
    return {
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.01, 0.03, 0.05, 0.1, 0.2]
        ),
        "warmup_steps": trial.suggest_categorical(
            "warmup_steps", [20, 40, 50, 60, 100, 200]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 15, 25, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64]
        ),
    }
