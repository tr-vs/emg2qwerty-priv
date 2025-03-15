#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize Optuna hyperparameter search results")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the study to visualize",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Storage URL for the study (e.g., sqlite:///optuna_studies/study_name.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_visualizations",
        help="Directory to save visualization plots",
    )
    args = parser.parse_args()

    # If storage is not provided, try to find it based on study name
    if args.storage is None and args.study_name is not None:
        storage_name = f"sqlite:///{os.getcwd()}/optuna_studies/{args.study_name}.db"
        if os.path.exists(storage_name.replace("sqlite:///", "")):
            args.storage = storage_name
        else:
            print(f"Could not find storage at {storage_name.replace('sqlite:///', '')}")
            print("Please provide a storage URL using --storage")
            return

    # If study name is not provided, try to extract it from storage
    if args.study_name is None and args.storage is not None:
        if "sqlite:///" in args.storage:
            db_path = args.storage.replace("sqlite:///", "")
            args.study_name = os.path.basename(db_path).replace(".db", "")

    if args.storage is None:
        print("Please provide either --study-name or --storage")
        return

    print(args.study_name)
    print(args.storage)
    # Load the study
    args.study_name = "emg2qwerty_optuna_HybridModule"
    args.storage = "sqlite:////emg2qwerty-priv/optuna_studies/emg2qwerty_optuna_HybridModule.db"
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    
    print(f"Loaded study '{args.study_name}' with {len(study.trials)} trials")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.study_name:
        output_dir = output_dir / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save visualizations
    visualizations = {
        "optimization_history": plot_optimization_history(study),
        "param_importances": plot_param_importances(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
        "contour": plot_contour(study),
        "slice": plot_slice(study),
        "edf": plot_edf(study),
    }

    # Check if there are intermediate values to plot
    if any(len(t.intermediate_values) > 0 for t in study.trials):
        visualizations["intermediate_values"] = plot_intermediate_values(study)

    # Save all visualizations
    for name, fig in visualizations.items():
        output_path = output_dir / f"{name}.html"
        fig.write_html(str(output_path))
        print(f"Saved {name} plot to {output_path}")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()