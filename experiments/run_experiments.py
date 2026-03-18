"""
Experiment Runner - Run multiple experiments for hyperparameter tuning.

This script runs multiple experiments with different configurations
and logs all results to MLflow for comparison.

TODO: Complete the functions marked with TODO.

Usage:
    python -m experiments.run_experiments
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from pipeline.config import EXPERIMENT_CONFIGS, MLFLOW_EXPERIMENT_NAME
from pipeline.data_ingestion import load_and_split
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs
from pipeline.training import setup_mlflow, train_model

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHART_OUTPUT_DIR = Path("artifacts/experiment_charts")


# =============================================================================
# TODO 1: Implement run_single_experiment function
# =============================================================================
def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = "hyperparameter-tuning",
) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.

    TODO: Implement this function that:
    1. Sets the MLflow experiment
    2. Extracts model_type from config
    3. Trains the model using train_model()
    4. Evaluates the model using evaluate_model()
    5. Returns experiment results

    Args:
        trainset: Training data
        testset: Test data
        config: Configuration dictionary with model_type and hyperparameters
        experiment_name: Name of the MLflow experiment

    Returns:
        Dictionary with experiment results:
        {
            'config': dict,
            'run_id': str,
            'metrics': dict
        }
    """
    mlflow.set_experiment(experiment_name)

    config_copy = config.copy()
    model_type = config_copy.pop("model_type")

    run_name_parts = [
        f"{key}={value}"
        for key, value in config_copy.items()
        if not isinstance(value, dict)
    ]
    run_name = f"{model_type}_{'_'.join(run_name_parts)}" if run_name_parts else model_type

    model, run_id = train_model(
        trainset,
        model_type=model_type,
        run_name=run_name,
        **config_copy,
    )
    metrics = evaluate_model(model, testset, run_id)

    return {
        "config": config.copy(),
        "run_id": run_id,
        "metrics": metrics,
    }


# =============================================================================
# TODO 2: Implement run_all_experiments function
# =============================================================================
def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = "hyperparameter-tuning",
) -> List[Dict[str, Any]]:
    """
    Run all experiments defined in configs.

    TODO: Implement this function that:
    1. Loads data once (for efficiency)
    2. Iterates through all configs
    3. Runs each experiment using run_single_experiment()
    4. Collects and returns all results

    Args:
        configs: List of configuration dictionaries
        experiment_name: Name of the MLflow experiment

    Returns:
        List of experiment results
    """
    logger.info(f"Running {len(configs)} experiments...")

    trainset, testset, _ = load_and_split()

    results = []
    for i, config in enumerate(configs, start=1):
        logger.info(f"\nExperiment {i}/{len(configs)}: {config}")
        try:
            result = run_single_experiment(
                trainset,
                testset,
                config,
                experiment_name=experiment_name,
            )
            results.append(result)
            logger.info(f"  RMSE: {result['metrics']['rmse']:.4f}")
            logger.info(f"  MAE: {result['metrics']['mae']:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"config": config.copy(), "error": str(e)})

    return results


# =============================================================================
# TODO 3: Implement generate_experiment_report function
# =============================================================================
def generate_experiment_report(
    results: List[Dict[str, Any]], output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a markdown report from experiment results.

    TODO: Implement this function that creates a markdown report with:
    1. Summary statistics
    2. Table of all experiments with metrics
    3. Best performing model details
    4. Recommendations

    Args:
        results: List of experiment results
        output_path: Path to save the report

    Returns:
        Report content as string
    """
    report = []
    report.append("# Experiment Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    successful = [r for r in results if "metrics" in r]
    failed = [r for r in results if "error" in r]
    visualizations = generate_experiment_visualizations(successful)

    report.append("## Summary\n")
    report.append(f"- Total experiments: {len(results)}")
    report.append(f"- Successful: {len(successful)}")
    report.append(f"- Failed: {len(failed)}\n")

    report.append("## Results\n")
    report.append("| Model | Parameters | RMSE | MAE | Run ID |")
    report.append("|-------|------------|------|-----|--------|")

    for result in successful:
        model_type = result["config"].get("model_type", "unknown")
        params = {
            key: value
            for key, value in result["config"].items()
            if key != "model_type"
        }
        report.append(
            f"| {model_type} | {json.dumps(params, sort_keys=True)} | "
            f"{result['metrics']['rmse']:.4f} | {result['metrics']['mae']:.4f} | "
            f"{result['run_id']} |"
        )

    if failed:
        report.append("\n## Failed Experiments\n")
        for result in failed:
            report.append(
                f"- `{json.dumps(result['config'], sort_keys=True)}`: {result['error']}"
            )

    if successful:
        best = min(successful, key=lambda x: x["metrics"].get("rmse", float("inf")))
        report.append("\n## Best Model\n")
        report.append(f"- Configuration: `{json.dumps(best['config'], sort_keys=True)}`")
        report.append(f"- RMSE: {best['metrics']['rmse']:.4f}")
        report.append(f"- MAE: {best['metrics']['mae']:.4f}")
        report.append(f"- Run ID: `{best['run_id']}`")

        if visualizations:
            report.append("\n## Visualizations\n")
            if visualizations.get("rmse_bar_chart"):
                report.append("### RMSE Across All Runs\n")
                report.append(
                    f"![RMSE across all runs]({visualizations['rmse_bar_chart']})"
                )
            if visualizations.get("rmse_mae_scatter"):
                report.append("\n### RMSE vs MAE Comparison\n")
                report.append(
                    f"![RMSE vs MAE comparison]({visualizations['rmse_mae_scatter']})"
                )
            if visualizations.get("best_model_diagnostic"):
                report.append("\n### Best Model Diagnostic Plot\n")
                report.append(
                    f"![Best model prediction distribution]({visualizations['best_model_diagnostic']})"
                )

        report.append("\n## Recommendations\n")
        report.append(
            f"- Select `{best['config'].get('model_type', 'unknown')}` with run ID "
            f"`{best['run_id']}` for production because it achieved the best RMSE "
            f"({best['metrics']['rmse']:.4f}) while maintaining strong MAE "
            f"({best['metrics']['mae']:.4f})."
        )
        report.append(
            "- Use the comparison charts above to justify why the selected configuration "
            "outperformed the other experiment runs."
        )
        report.append(
            "- Keep the prediction distribution plot in the report as the diagnostic "
            "visualization for the chosen production model."
        )
    else:
        report.append("\n## Recommendations\n")
        report.append("- No successful experiments were available. Fix the failed runs and rerun the sweep.")

    content = "\n".join(report) + "\n"

    with open(output_path, "w") as f:
        f.write(content)

    return content


def generate_experiment_visualizations(
    successful_results: List[Dict[str, Any]],
    output_dir: Path = CHART_OUTPUT_DIR,
) -> Dict[str, str]:
    """Create report-ready experiment comparison charts and diagnostic plots."""
    if not successful_results:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        "svd": "#1f77b4",
        "nmf": "#ff7f0e",
        "knn": "#2ca02c",
    }

    labels = [
        f"{result['config'].get('model_type', 'model')}-{index}"
        for index, result in enumerate(successful_results, start=1)
    ]
    rmse_values = [result["metrics"]["rmse"] for result in successful_results]
    mae_values = [result["metrics"]["mae"] for result in successful_results]
    colors = [
        color_map.get(result["config"].get("model_type", "unknown"), "#7f7f7f")
        for result in successful_results
    ]

    rmse_chart_path = output_dir / "rmse_across_runs.png"
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, rmse_values, color=colors)
    plt.ylabel("RMSE")
    plt.xlabel("Experiment Run")
    plt.title("RMSE Across All Experiment Runs")
    plt.xticks(rotation=35, ha="right")
    for bar, value in zip(bars, rmse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(rmse_chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    scatter_chart_path = output_dir / "rmse_vs_mae_scatter.png"
    plt.figure(figsize=(10, 6))
    for label, rmse, mae, color, result in zip(
        labels, rmse_values, mae_values, colors, successful_results
    ):
        plt.scatter(rmse, mae, s=90, color=color)
        plt.annotate(label, (rmse, mae), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("RMSE")
    plt.ylabel("MAE")
    plt.title("RMSE vs MAE Across Experiment Runs")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(scatter_chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    best_result = min(
        successful_results,
        key=lambda result: result["metrics"].get("rmse", float("inf")),
    )
    diagnostic_chart_path = download_best_model_diagnostic(
        best_result["run_id"],
        output_dir,
    )

    visualizations = {
        "rmse_bar_chart": str(rmse_chart_path),
        "rmse_mae_scatter": str(scatter_chart_path),
    }
    if diagnostic_chart_path is not None:
        visualizations["best_model_diagnostic"] = str(diagnostic_chart_path)
    return visualizations


def download_best_model_diagnostic(
    run_id: str, output_dir: Path
) -> Optional[Path]:
    """Download the best run's prediction distribution plot from MLflow."""
    client = MlflowClient()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = Path(
            client.download_artifacts(
                run_id,
                "prediction_distribution.png",
                str(output_dir),
            )
        )
    except Exception as exc:
        logger.warning(
            "Could not download prediction_distribution.png for run %s: %s",
            run_id,
            exc,
        )
        return None

    target_path = output_dir / "best_model_prediction_distribution.png"
    if downloaded_path != target_path:
        shutil.copy2(downloaded_path, target_path)
    return target_path


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Run all experiments and generate report."""

    logger.info("=" * 60)
    logger.info("Starting Experiment Runner")
    logger.info("=" * 60)

    # Setup MLflow
    setup_mlflow()

    # Run experiments
    results = run_all_experiments(
        configs=EXPERIMENT_CONFIGS, experiment_name="hyperparameter-tuning"
    )

    # Generate report
    report = generate_experiment_report(results, "experiment_report.md")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Summary")
    logger.info("=" * 60)

    successful = [r for r in results if "metrics" in r]
    if successful:
        best = min(successful, key=lambda x: x["metrics"].get("rmse", float("inf")))
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Best RMSE: {best['metrics']['rmse']:.4f}")
        logger.info(f"Best config: {best['config']}")

    # Compare top runs
    logger.info("\nTop 5 runs:")
    top_runs = compare_runs(
        experiment_name="hyperparameter-tuning",
        metric="rmse",
        top_n=5,
    )
    for i, run in enumerate(top_runs, 1):
        rmse = run["metrics"].get("rmse")
        rmse_display = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "N/A"
        logger.info(
            f"  {i}. RMSE={rmse_display} - {run['params']}"
        )

    logger.info(f"\nReport saved to: experiment_report.md")
    logger.info("View experiments in MLflow UI: http://localhost:5019")

    return results


if __name__ == "__main__":
    main()
