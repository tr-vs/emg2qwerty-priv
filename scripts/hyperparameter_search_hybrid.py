#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from emg2qwerty.train import main as train_main


log = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: DictConfig) -> float:
    """
    Objective function for Optuna to optimize.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration to modify
    
    Returns:
        Metric value to minimize/maximize
    """
    # Create a copy of the base config to modify
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

    if trial.number == 0:
        log.info("Running first trial with initial parameters from config")
        # Don't modify any parameters, use exactly what's in the config file
        result = train_main(config)
        baseline_score = result["metric_value"]
        trial.study.set_user_attr("baseline_score", baseline_score)
        log.info(f"Baseline score with initial parameters: {baseline_score}")
        return baseline_score
    
    # Common hyperparameters
    # config.optimizer.lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2])
    
    # Number of blocks from 1 to 4
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    config.module.block_channels = [24] * num_blocks
    
    # Kernel width - just try 15 or 31
    config.module.kernel_width = trial.suggest_categorical("kernel_width", [16, 32, 48])
    
    # RNN hidden size
    config.module.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    
    # RNN number of layers - only 2 or 4
    config.module.num_layers = trial.suggest_categorical("num_layers", [2, 4])

    # Run training with the modified config
    result = train_main(config)
    
    # Return the metric to optimize (assuming lower is better)
    metric_value = result["metric_value"]
    
    # Compare against baseline (optional)
    baseline_score = trial.study.user_attrs["baseline_score"]
    if metric_value >= baseline_score:
        raise optuna.TrialPruned(f"Score {metric_value} is not better than baseline {baseline_score}")

    return metric_value


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        config: Base configuration
    """
    log.info(f"Starting hyperparameter optimization with Optuna")
    log.info(f"Base config:\n{OmegaConf.to_yaml(config)}")
    
    # Create a study object
    direction = "minimize" if config.monitor_mode == "min" else "maximize"
    study_name = f"emg2qwerty_optuna_{config.module._target_.split('.')[-1]}"
    
    # Create database for storing results
    storage_name = f"sqlite:///{os.getcwd()}/optuna_studies/{study_name}.db"
    os.makedirs(os.path.dirname(storage_name), exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction=direction,
        load_if_exists=True,
    )
    
    # Set the number of trials
    n_trials = config.get("optuna", {}).get("n_trials", 20)
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=n_trials,
        timeout=config.get("optuna", {}).get("timeout", None),
        show_progress_bar=True,
    )
    
    # Print the results
    log.info(f"Best trial:")
    best_trial = study.best_trial
    log.info(f"  Value: {best_trial.value}")
    log.info(f"  Params:")
    for key, value in best_trial.params.items():
        log.info(f"    {key}: {value}")
    
    # Create a config with the best parameters
    best_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # Update the config with the best parameters
    for key, value in best_trial.params.items():
        if key == "lr":
            best_config.optimizer.lr = value
        elif key == "batch_size":
            best_config.batch_size = value
        elif key in ["block_channels", "kernel_width", "hidden_size", "num_layers"] and "HybridModule" in best_config.module._target_.lower():
            # Add hybrid specific parameters
            OmegaConf.update(best_config, f"module.{key}", value)
    
    # Save the best config
    output_dir = Path(os.getcwd()) / "best_configs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{study_name}_best_config.yaml"
    
    with open(output_path, "w") as f:
        f.write(OmegaConf.to_yaml(best_config))
    
    log.info(f"Best configuration saved to {output_path}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", lambda: os.cpu_count())
    main() 