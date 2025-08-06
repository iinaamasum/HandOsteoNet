import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


def setup_plotting_style():
    """Setup professional plotting style"""
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "#F4F7FD"
    plt.rcParams["axes.facecolor"] = "#FAFCFF"
    plt.rcParams["grid.color"] = "#E4EAED"

    COLORS = {
        "blue": "#1f77b4",
        "green": "#2ca02c",
        "orange": "#ff7f0e",
        "gray": "#7f7f7f",
    }
    return COLORS


def generate_regression_plots(
    predictions_csv="predictions.csv",
    metrics_csv="test_metrics.csv",
    history_csv="training_metrics.csv",
    output_dir="plots",
):
    """
    Generate comprehensive evaluation plots

    Args:
        predictions_csv: Path to predictions CSV
        metrics_csv: Path to metrics CSV
        history_csv: Path to training history CSV
        output_dir: Output directory for plots
    """
    setup_plotting_style()
    COLORS = setup_plotting_style()

    os.makedirs(output_dir, exist_ok=True)

    try:
        results_df = pd.read_csv(predictions_csv)
        predictions = results_df["predictions"].values
        actuals = results_df["actuals"].values
        metrics_df = pd.read_csv(metrics_csv)
        mae = metrics_df["mae_mean"].iloc[0]
        mae_ci = metrics_df["mae_ci"].iloc[0]
        rmse = metrics_df["rmse_mean"].iloc[0]
        rmse_ci = metrics_df["rmse_ci"].iloc[0]
        mape = metrics_df["mape_mean"].iloc[0]
        mape_ci = metrics_df["mape_ci"].iloc[0]
        r2 = metrics_df["r2"].iloc[0]
        medae = metrics_df["medae"].iloc[0]
        medae_ci = metrics_df["medae_ci"].iloc[0]
    except FileNotFoundError:
        print("Input files not found. Using synthetic data for demonstration.")
        return

    try:
        history_df = pd.read_csv(history_csv)
        epochs = history_df["epoch"].values
        train_losses = history_df["train_loss"].values
        val_losses = history_df["val_loss"].values
        train_maes = history_df["train_mae"].values
        val_maes = history_df["val_mae"].values
    except FileNotFoundError:
        print("Training history file not found. Using synthetic training data.")
        return

    residuals = predictions - actuals
    abs_errors = np.abs(residuals)

    # Scatter Plot: Predictions vs. Actuals
    plt.figure()
    sns.scatterplot(x=actuals, y=predictions, color=COLORS["blue"], alpha=0.7, s=40)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLORS["orange"],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction (y=x)",
    )
    plt.xlabel("Actual Bone Age (months)")
    plt.ylabel("Predicted Bone Age (months)")
    plt.title("Predictions vs. Actual Bone Ages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_plot.pdf"))
    plt.close()

    # Residual Plot
    plt.figure()
    sns.scatterplot(x=predictions, y=residuals, color=COLORS["green"], alpha=0.7, s=40)
    plt.axhline(
        0, color=COLORS["orange"], linestyle="--", linewidth=2, label="Zero Line"
    )
    plt.xlabel("Predicted Bone Age (months)")
    plt.ylabel("Residuals (months)")
    plt.title("Residuals vs. Predicted Bone Ages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_plot.pdf"))
    plt.close()

    # Histogram: Residuals
    plt.figure()
    sns.histplot(residuals, bins=30, color=COLORS["blue"], kde=True, edgecolor="black")
    plt.xlabel("Residuals (months)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_histogram.pdf"))
    plt.close()

    # Q-Q Plot: Residual Normality
    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles (Residuals)")
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True, color=COLORS["gray"], linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot.pdf"))
    plt.close()

    # Prediction Error Plot
    plt.figure()
    sns.scatterplot(x=actuals, y=abs_errors, color=COLORS["green"], alpha=0.7, s=40)
    plt.axhline(
        mae,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=2,
        label=f"MAE = {mae:.2f} months",
    )
    plt.xlabel("Actual Bone Age (months)")
    plt.ylabel("Absolute Error (months)")
    plt.title("Absolute Prediction Errors vs. Actual Bone Ages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_error_plot.pdf"))
    plt.close()

    # Cumulative Error Distribution (CDF) Plot
    plt.figure()
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.step(sorted_errors, cdf, color=COLORS["blue"], linewidth=2, label="CDF")
    plt.axvline(
        mae,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=2,
        label=f"MAE = {mae:.2f} months",
    )
    plt.axvline(
        medae,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label=f"MedAE = {medae:.2f} months",
    )
    plt.xlabel("Absolute Error (months)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Absolute Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf_error_plot.pdf"))
    plt.close()

    # Training and Validation Loss Plot
    plt.figure()
    plt.plot(
        epochs,
        train_losses,
        "-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Training Loss",
    )
    plt.plot(
        epochs,
        val_losses,
        "-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.pdf"))
    plt.close()

    # Training and Validation MAE Plot
    plt.figure()
    plt.plot(
        epochs,
        train_maes,
        "-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Training MAE",
    )
    plt.plot(
        epochs,
        val_maes,
        "-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Validation MAE",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (months)")
    plt.title("Training and Validation MAE Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_curve.pdf"))
    plt.close()

    print(f"All plots saved to '{output_dir}' directory.")


if __name__ == "__main__":
    generate_regression_plots()
