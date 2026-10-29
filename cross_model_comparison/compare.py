"""
Cross-Model Comparison Script
Matches 8_Cross_Model_Comparison.ipynb.
Aggregates metrics from individual baseline runs in results/,
generates consolidated CSV/JSON reports, and plots comparison visualizations.
Outputs saved to results/cross_model_comparison/ and logs to logs/comparison.log.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_INFO = [
    ("1_B1_Image_Classification", "Model 1: B1 - Image Classification"),
    ("2_B3_Finetuned_Person_Classification", "Model 2: B3 - Fine-tuned Person Classification"),
    ("3_B4_Temporal_Model_Image_Features", "Model 3: B4 - Temporal Model with Image Features"),
    ("4_B5_Temporal_Model_Person_Features", "Model 4: B5 - Temporal Model with Person Features"),
    ("5_B6_Two_Stage_No_LSTM1", "Model 5: B6 - Two-stage Model without LSTM 1"),
    ("6_B7_Two_Stage_No_LSTM2", "Model 6: B7 - Two-stage Model without LSTM 2"),
    ("7_Model7_Hierarchical_Full", "Model 7: Two-stage Hierarchical Model (Full)")
]


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("CrossModelComp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def run_cross_model_comparison(
    results_base: str = "results",
    output_dir: str = "results/cross_model_comparison"
):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "comparison.log"))

    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    records = []
    logger.info("Aggregating metrics across all baseline directories...")

    for folder_name, display_name in MODELS_INFO:
        metrics_path = os.path.join(results_base, folder_name, "metrics", "metrics.json")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    data["Model"] = display_name
                    data["Folder"] = folder_name
                    records.append(data)
                    logger.info(f"Loaded metrics for {display_name}")
            except Exception as e:
                logger.warning(f"Error loading {metrics_path}: {e}")
        else:
            records.append({
                "Model": display_name,
                "Folder": folder_name,
                "test_accuracy": np.nan,
                "test_f1_weighted": np.nan,
                "test_f1_macro": np.nan,
                "test_precision": np.nan,
                "test_recall": np.nan,
                "best_val_accuracy": np.nan,
                "status": "Pending / Not Run"
            })
            logger.info(f"Metrics not found for {display_name} ({metrics_path})")

    df_comparison = pd.DataFrame(records)

    # Format table for logging
    cols_to_display = [c for c in ['Model', 'test_accuracy', 'test_f1_weighted', 'test_f1_macro', 'status'] if c in df_comparison.columns]
    logger.info("\n=== Consolidated Model Comparison Table ===")
    logger.info("\n" + df_comparison[cols_to_display].to_string(index=False))

    # Save to CSV and JSON
    csv_path = os.path.join(metrics_dir, "cross_model_metrics.csv")
    json_path = os.path.join(metrics_dir, "cross_model_metrics.json")
    df_comparison.to_csv(csv_path, index=False)
    df_comparison.to_json(json_path, orient="records", indent=4)
    logger.info(f"Saved consolidated metrics to {csv_path} and {json_path}")

    # Generate visual comparison plot
    valid_df = df_comparison.dropna(subset=['test_accuracy'])
    if not valid_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(valid_df['Model'], valid_df['test_accuracy'] * 100, color='royalblue', edgecolor='navy')
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title('Cross-Model Comparison: Group Activity Recognition Accuracy', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved accuracy comparison plot to {plot_path}")
    else:
        logger.info("No completed test accuracies found yet. Visualization will be plotted once models are evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Model Comparison Aggregator")
    parser.add_argument("--results_base", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/cross_model_comparison")
    args = parser.parse_args()

    run_cross_model_comparison(
        results_base=args.results_base,
        output_dir=args.output_dir
    )
