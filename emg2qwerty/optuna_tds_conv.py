import os
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Import your training function
from emg2qwerty.train import main as train_main


def objective(trial: optuna.Trial, config: DictConfig) -> float:
    """Optuna objective function to minimize for TDSConvCTCModule.
    
    Args:
        trial: Optuna trial object
        config: Base hydra config
    
    Returns:
        Validation character error rate (CER) to minimize
    """
    # TDSConvCTCModule specific hyperparameters
    
    # 1. MLP features - controls the size of the MLP layers
    mlp_dim = trial.suggest_int("mlp_dim", 128, 512, step=64)
    config.module.mlp_features = [mlp_dim]
    
    # 2. Block channels - controls the number of channels in each TDS block
    # We can either use the same number for all blocks or different numbers
    use_uniform_channels = trial.suggest_categorical("use_uniform_channels", [True, False])
    
    if use_uniform_channels:
        channels = trial.suggest_int("uniform_channels", 16, 64, step=8)
        num_blocks = trial.suggest_int("num_blocks", 2, 6)
        config.module.block_channels = [channels] * num_blocks
    else:
        # Different channel counts for each block, with increasing size
        num_blocks = trial.suggest_int("num_blocks", 2, 6)
        channels = []
        for i in range(num_blocks):
            # Suggest larger channel ranges for deeper blocks
            min_ch = 16 + i * 4
            max_ch = 32 + i * 8
            ch = trial.suggest_int(f"block_{i}_channels", min_ch, max_ch, step=8)
            channels.append(ch)
        config.module.block_channels = channels
    
    # 3. Kernel width - controls the temporal receptive field
    kernel_width = trial.suggest_int("kernel_width", 16, 64, step=8)
    config.module.kernel_width = kernel_width
    
    # 4. Learning rate and optimizer parameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config.optimizer.lr = lr
    
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    if "weight_decay" in config.optimizer:
        config.optimizer.weight_decay = weight_decay
    
    # 5. Dropout rate (if applicable)
    # Note: You might need to add this parameter to your model if not already present
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    if hasattr(config.module, "dropout"):
        config.module.dropout = dropout
    
    # 6. Data processing parameters
    # Window length and padding can significantly affect performance
    window_length = trial.suggest_int("window_length", 4000, 12000, step=2000)
    config.datamodule.window_length = window_length
    
    left_padding = trial.suggest_int("left_padding", 800, 2400, step=400)
    right_padding = trial.suggest_int("right_padding", 100, 500, step=100)
    config.datamodule.padding = [left_padding, right_padding]
    
    # Set a unique output directory for this trial
    trial_dir = Path(config.hydra.run.dir) / f"trial_{trial.number}"
    config.hydra.run.dir = str(trial_dir)
    
    # Add trial info to the config
    config.trial = {
        "number": trial.number,
        "params": OmegaConf.create(trial.params)
    }
    
    # Run training with these hyperparameters
    try:
        # Reduce number of epochs for faster trials
        if "max_epochs" in config.trainer:
            config.trainer.max_epochs = min(config.trainer.max_epochs, 20)
        
        # Run training
        result = train_main(config)
        
        # Return the validation metric to minimize
        return result.get("best_val_cer", float('inf'))
    except Exception as e:
        # Log the error but don't fail the entire study
        logging.error(f"Trial {trial.number} failed with error: {e}")
        # Return a high value to indicate failure
        return float('inf')


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run_optuna_study(config: DictConfig) -> Dict[str, Any]:
    """Run an Optuna hyperparameter search for TDSConvCTCModule."""
    # Create study directory
    study_name = config.get("study_name", "tds_conv_ctc_study")
    study_dir = Path(config.get("study_dir", "./optuna_studies"))
    study_dir.mkdir(exist_ok=True, parents=True)
    
    storage_name = f"sqlite:///{study_dir}/{study_name}.db"
    
    # Create or load the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",  # Minimize character error rate
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run the optimization
    n_trials = config.get("n_trials", 20)
    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=n_trials
    )
    
    # Print results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best configuration
    best_config = OmegaConf.create(config)
    
    # Update best config with best parameters
    if best_trial.params.get("use_uniform_channels", True):
        channels = best_trial.params.get("uniform_channels", 24)
        num_blocks = best_trial.params.get("num_blocks", 4)
        best_config.module.block_channels = [channels] * num_blocks
    else:
        channels = []
        num_blocks = best_trial.params.get("num_blocks", 4)
        for i in range(num_blocks):
            ch = best_trial.params.get(f"block_{i}_channels", 24)
            channels.append(ch)
        best_config.module.block_channels = channels
    
    best_config.module.mlp_features = [best_trial.params.get("mlp_dim", 384)]
    best_config.module.kernel_width = best_trial.params.get("kernel_width", 32)
    best_config.optimizer.lr = best_trial.params.get("learning_rate", 1e-3)
    
    if "weight_decay" in best_config.optimizer:
        best_config.optimizer.weight_decay = best_trial.params.get("weight_decay", 0)
    
    if hasattr(best_config.module, "dropout"):
        best_config.module.dropout = best_trial.params.get("dropout", 0.1)
    
    best_config.datamodule.window_length = best_trial.params.get("window_length", 8000)
    best_config.datamodule.padding = [
        best_trial.params.get("left_padding", 1800),
        best_trial.params.get("right_padding", 200)
    ]
    
    # Save the best configuration
    best_config_path = study_dir / f"{study_name}_best_config.yaml"
    with open(best_config_path, "w") as f:
        OmegaConf.save(best_config, f)
    
    # Create a visualization of the optimization results
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(study_dir / f"{study_name}_history.png"))
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(str(study_dir / f"{study_name}_importance.png"))
        
        print(f"Visualizations saved to {study_dir}")
    except ImportError:
        print("Could not create visualizations. Make sure matplotlib and plotly are installed.")
    
    return {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "best_config_path": str(best_config_path)
    }


if __name__ == "__main__":
    run_optuna_study() 