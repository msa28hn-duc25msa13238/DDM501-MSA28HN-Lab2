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
from datetime import datetime
from typing import Any, Dict, List

import mlflow

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

        report.append("\n## Recommendations\n")
        report.append(
            f"- Promote the `{best['config'].get('model_type', 'unknown')}` configuration "
            f"with the lowest RMSE ({best['metrics']['rmse']:.4f}) for further validation."
        )
        report.append(
            "- Review MLflow artifacts and prediction plots for the top runs before registration."
        )
    else:
        report.append("\n## Recommendations\n")
        report.append("- No successful experiments were available. Fix the failed runs and rerun the sweep.")

    content = "\n".join(report) + "\n"

    with open(output_path, "w") as f:
        f.write(content)

    return content


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
    top_runs = compare_runs(metric="rmse", top_n=5)
    for i, run in enumerate(top_runs, 1):
        logger.info(
            f"  {i}. RMSE={run['metrics'].get('rmse', 'N/A'):.4f} - {run['params']}"
        )

    logger.info(f"\nReport saved to: experiment_report.md")
    logger.info("View experiments in MLflow UI: http://localhost:5019")

    return results


if __name__ == "__main__":
    main()
