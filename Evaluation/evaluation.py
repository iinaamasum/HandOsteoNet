import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..Training.loss_function import CombinedLoss


def bootstrap_ci(data, stat_func, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals

    Args:
        data: Data to bootstrap
        stat_func: Function to calculate statistic
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        delta: Half-width of confidence interval
    """
    stats = [
        stat_func(np.random.choice(data, len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    lower, upper = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta = (upper - lower) / 2
    return delta


def evaluate_model(
    model, test_loader, device, model_path_eval, output_csv="test_metrics.csv"
):
    """
    Evaluate the HandOsteoNet model on test data

    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        model_path_eval: Path to model weights
        output_csv: Path to save metrics CSV

    Returns:
        metrics_df: DataFrame with evaluation metrics
        results_df: DataFrame with predictions and actuals
    """
    criterion = CombinedLoss(alpha=0.8, beta=0.1, gamma=0.1)
    model.eval()
    model.load_state_dict(torch.load(model_path_eval, map_location=device))

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, boneages, males, _ in tqdm(test_loader, desc="Evaluating Test"):
            images = images.to(device)
            boneages = boneages.to(device)
            males = males.to(device)

            outputs = model(images, males).squeeze(-1)
            all_predictions.append(outputs.cpu())
            all_targets.append(boneages.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mae = np.mean(np.abs(all_predictions - all_targets))
    mae_ci = bootstrap_ci(np.abs(all_predictions - all_targets), np.mean)

    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    rmse_ci = bootstrap_ci(
        (all_predictions - all_targets) ** 2, lambda x: np.sqrt(np.mean(x))
    )

    mape = (
        np.mean(np.abs((all_predictions - all_targets) / (all_targets + 1e-10))) * 100
    )
    mape_ci = bootstrap_ci(
        np.abs((all_predictions - all_targets) / (all_targets + 1e-10)) * 100, np.mean
    )

    target_mean = np.mean(all_targets)
    ss_tot = np.sum((all_targets - target_mean) ** 2)
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    medae = np.median(np.abs(all_predictions - all_targets))
    medae_ci = bootstrap_ci(np.abs(all_predictions - all_targets), np.median)

    metrics_df = pd.DataFrame(
        {
            "split": ["test"],
            "mae_mean": [mae],
            "mae_ci": [mae_ci],
            "rmse_mean": [rmse],
            "rmse_ci": [rmse_ci],
            "mape_mean": [mape],
            "mape_ci": [mape_ci],
            "r2": [r2],
            "medae": [medae],
            "medae_ci": [medae_ci],
        }
    )
    metrics_df.to_csv(output_csv, index=False)

    results_df = pd.DataFrame({"predictions": all_predictions, "actuals": all_targets})
    results_df.to_csv("predictions.csv", index=False)

    print(f"\nTest Metrics:")
    print(f"MAE:           {mae:.4f} ± {mae_ci:.4f} months")
    print(f"MAPE:          {mape:.4f} ± {mape_ci:.4f} %")
    print(f"R²:            {r2:.4f}")
    print(f"MedAE:         {medae:.4f} ± {medae_ci:.4f} months")

    return metrics_df, results_df
