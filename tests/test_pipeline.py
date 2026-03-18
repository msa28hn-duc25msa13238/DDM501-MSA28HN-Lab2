"""
Unit tests for ML Pipeline.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


class DummyPrediction:
    """Minimal Surprise-style prediction object for tests."""

    def __init__(self, est=4.0):
        self.est = est


class DummyModel:
    """Pickle-friendly fake model used for MLflow logging tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_on = None

    def fit(self, trainset):
        self.fitted_on = trainset
        return self

    def predict(self, user_id, item_id):
        return DummyPrediction(est=4.0)


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_load_data_returns_dataset(self):
        """Test that load_data returns a dataset object."""
        from pipeline.data_ingestion import load_data
        
        data = load_data('ml-100k')
        assert data is not None

    def test_load_data_uses_non_interactive_builtin_download(self):
        """Dataset loading should not prompt for input in Airflow/docker contexts."""
        from pipeline.data_ingestion import load_data

        with patch("pipeline.data_ingestion.Dataset.load_builtin", return_value="dataset") as mock_load:
            data = load_data("ml-100k")

        assert data == "dataset"
        mock_load.assert_called_once_with("ml-100k", prompt=False)
    
    def test_split_data_returns_train_test(self):
        """Test that split_data returns trainset and testset."""
        from pipeline.data_ingestion import load_data, split_data
        
        data = load_data('ml-100k')
        trainset, testset = split_data(data, test_size=0.2)
        
        assert trainset is not None
        assert testset is not None
        assert len(testset) > 0
    
    def test_get_data_stats(self):
        """Test that get_data_stats returns correct statistics."""
        from pipeline.data_ingestion import load_data, get_data_stats
        
        data = load_data('ml-100k')
        stats = get_data_stats(data)
        
        assert 'n_users' in stats
        assert 'n_items' in stats
        assert 'n_ratings' in stats
        assert stats['n_users'] > 0
        assert stats['n_items'] > 0


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_validate_trainset(self):
        """Test trainset validation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import validate_trainset
        
        trainset, _, _ = load_and_split()
        report = validate_trainset(trainset)
        
        assert 'is_valid' in report
        assert report['is_valid'] == True
    
    def test_get_rating_distribution(self):
        """Test rating distribution calculation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import get_rating_distribution
        
        trainset, _, _ = load_and_split()
        dist = get_rating_distribution(trainset)
        
        assert 'mean' in dist
        assert 'std' in dist
        assert 1.0 <= dist['mean'] <= 5.0


class TestTraining:
    """Tests for training module."""
    
    def test_list_available_models(self):
        """Test that available models are listed."""
        from pipeline.training import list_available_models
        
        models = list_available_models()
        assert 'svd' in models
        assert 'nmf' in models
        assert 'knn' in models
    
    def test_get_default_params(self):
        """Test getting default parameters."""
        from pipeline.training import get_default_params
        
        params = get_default_params('svd')
        assert 'n_factors' in params
        assert 'n_epochs' in params

    def test_train_model_logs_registry_model(self, tmp_path):
        """Training should log both the pickle artifact and MLflow model artifact."""
        from pipeline.training import train_model

        active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        with patch("pipeline.training.MODELS_DIR", tmp_path), patch.dict(
            "pipeline.training.MODEL_CLASSES", {"svd": DummyModel}, clear=False
        ), patch("pipeline.training.mlflow.start_run") as mock_start_run, patch(
            "pipeline.training.mlflow.active_run", return_value=active_run
        ), patch("pipeline.training.mlflow.log_param") as mock_log_param, patch(
            "pipeline.training.mlflow.log_artifact"
        ) as mock_log_artifact, patch(
            "pipeline.training.mlflow.pyfunc.log_model"
        ) as mock_log_model:
            mock_start_run.return_value.__enter__.return_value = active_run
            mock_start_run.return_value.__exit__.return_value = False

            model, run_id = train_model(
                trainset="trainset",
                model_type="svd",
                n_factors=10,
                n_epochs=5,
            )

        assert isinstance(model, DummyModel)
        assert model.fitted_on == "trainset"
        assert run_id == "run-123"
        mock_log_param.assert_any_call("model_type", "svd")
        mock_log_param.assert_any_call("n_factors", 10)
        mock_log_param.assert_any_call("n_epochs", 5)
        mock_log_artifact.assert_called_once()

        logged_model_path = tmp_path / "model_svd_run-123.pkl"
        assert logged_model_path.exists()
        mock_log_model.assert_called_once()
        assert mock_log_model.call_args.kwargs["artifact_path"] == "model"
        assert (
            mock_log_model.call_args.kwargs["artifacts"]["model_pickle"]
            == str(logged_model_path)
        )
    
    # TODO: Add test for train_model after implementation
    # def test_train_model(self):
    #     """Test model training."""
    #     from pipeline.data_ingestion import load_and_split
    #     from pipeline.training import train_model, setup_mlflow
    #     
    #     setup_mlflow()
    #     trainset, _, _ = load_and_split()
    #     model, run_id = train_model(trainset, model_type='svd', n_factors=10, n_epochs=5)
    #     
    #     assert model is not None
    #     assert run_id is not None


class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_create_prediction_distribution_plot(self):
        """Test plot creation."""
        from pipeline.evaluation import create_prediction_distribution_plot
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split
        
        # Quick training for test
        data = Dataset.load_builtin('ml-100k')
        trainset, testset = train_test_split(data, test_size=0.1)
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)
        predictions = model.test(testset[:100])
        
        fig = create_prediction_distribution_plot(predictions)
        assert fig is not None


class TestRegistry:
    """Tests for registry module."""
    
    def test_list_registered_models(self):
        """Test listing registered models."""
        from pipeline.registry import list_registered_models
        
        # This should not raise an error
        models = list_registered_models()
        assert isinstance(models, list)

    def test_register_model_uses_mlflow_model_artifact(self):
        """Registry should register the MLflow model directory, not the raw pickle."""
        from pipeline.registry import register_model

        client = MagicMock()
        client.list_artifacts.return_value = [
            SimpleNamespace(path="model"),
            SimpleNamespace(path="model_svd_run-123.pkl"),
        ]
        result = SimpleNamespace(version="7")

        with patch("pipeline.registry.MlflowClient", return_value=client), patch(
            "pipeline.registry.mlflow.register_model", return_value=result
        ) as mock_register:
            version = register_model("run-123", "movie-rating-model")

        assert version == "7"
        mock_register.assert_called_once_with(
            "runs:/run-123/model", "movie-rating-model"
        )

    def test_register_best_model_returns_registration_summary(self):
        """Combined registry helper should return the selected run and version."""
        from pipeline.registry import register_best_model

        best_run = {
            "run_id": "run-123",
            "metrics": {"rmse": 0.91, "mae": 0.72},
            "params": {"model_type": "svd"},
            "artifact_uri": "mlruns:/artifact-uri",
        }

        with patch("pipeline.registry.find_best_run", return_value=best_run), patch(
            "pipeline.registry.register_model", return_value="3"
        ) as mock_register_model, patch(
            "pipeline.registry.transition_model_stage"
        ) as mock_transition:
            result = register_best_model(
                experiment_name="movie-rating-prediction",
                model_name="movie-rating-model",
                metric="rmse",
                stage="Production",
            )

        mock_register_model.assert_called_once_with("run-123", "movie-rating-model")
        mock_transition.assert_called_once_with(
            "movie-rating-model", "3", "Production"
        )
        assert result == {
            "run_id": "run-123",
            "model_name": "movie-rating-model",
            "version": "3",
            "stage": "Production",
            "metrics": {"rmse": 0.91, "mae": 0.72},
        }

    def test_find_best_run_skips_runs_without_model_artifact(self):
        """Best-run lookup should ignore legacy runs that cannot be registered."""
        from pipeline.registry import find_best_run

        client = MagicMock()
        client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="exp-1"
        )
        client.search_runs.return_value = [
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="legacy-best", artifact_uri="mlruns:/legacy-best"
                ),
                data=SimpleNamespace(metrics={"rmse": 0.90}, params={"model_type": "svd"}),
            ),
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="registerable-best",
                    artifact_uri="mlruns:/registerable-best",
                ),
                data=SimpleNamespace(metrics={"rmse": 0.94}, params={"model_type": "svd"}),
            ),
        ]
        client.list_artifacts.side_effect = [
            [SimpleNamespace(path="model_svd_legacy-best.pkl")],
            [SimpleNamespace(path="model"), SimpleNamespace(path="model_svd_registerable-best.pkl")],
        ]

        with patch("pipeline.registry.MlflowClient", return_value=client):
            best_run = find_best_run(
                experiment_name="movie-rating-prediction",
                metric="rmse",
                ascending=True,
                required_artifact_path="model",
            )

        assert best_run["run_id"] == "registerable-best"
        assert best_run["metrics"]["rmse"] == 0.94


class TestExperiments:
    """Tests for experiment sweep reporting helpers."""

    def test_generate_experiment_visualizations_creates_report_assets(self, tmp_path):
        """Experiment charts should be generated for RMSE/MAE and best-model diagnostics."""
        from experiments.run_experiments import generate_experiment_visualizations

        results = [
            {
                "config": {"model_type": "svd", "n_factors": 50},
                "run_id": "run-1",
                "metrics": {"rmse": 0.94, "mae": 0.74},
            },
            {
                "config": {"model_type": "nmf", "n_factors": 30},
                "run_id": "run-2",
                "metrics": {"rmse": 0.91, "mae": 0.72},
            },
        ]

        diagnostic_source = tmp_path / "prediction_distribution.png"
        diagnostic_source.write_bytes(b"fake-image")

        with patch(
            "experiments.run_experiments.download_best_model_diagnostic",
            return_value=diagnostic_source,
        ) as mock_download:
            visualizations = generate_experiment_visualizations(
                results, output_dir=tmp_path / "charts"
            )

        assert (tmp_path / "charts" / "rmse_across_runs.png").exists()
        assert (tmp_path / "charts" / "rmse_vs_mae_scatter.png").exists()
        assert visualizations["rmse_bar_chart"].endswith("rmse_across_runs.png")
        assert visualizations["rmse_mae_scatter"].endswith("rmse_vs_mae_scatter.png")
        assert visualizations["best_model_diagnostic"].endswith(
            "prediction_distribution.png"
        )
        mock_download.assert_called_once_with("run-2", tmp_path / "charts")

    def test_generate_experiment_report_includes_visualization_section(self, tmp_path):
        """Markdown report should include chart and production recommendation sections."""
        from experiments.run_experiments import generate_experiment_report

        results = [
            {
                "config": {"model_type": "svd", "n_factors": 50},
                "run_id": "run-1",
                "metrics": {"rmse": 0.93, "mae": 0.73},
            }
        ]
        report_path = tmp_path / "experiment_report.md"

        with patch(
            "experiments.run_experiments.generate_experiment_visualizations",
            return_value={
                "rmse_bar_chart": "artifacts/experiment_charts/rmse_across_runs.png",
                "rmse_mae_scatter": "artifacts/experiment_charts/rmse_vs_mae_scatter.png",
                "best_model_diagnostic": (
                    "artifacts/experiment_charts/"
                    "best_model_prediction_distribution.png"
                ),
            },
        ):
            report = generate_experiment_report(results, output_path=str(report_path))

        assert "## Visualizations" in report
        assert "## Recommendations" in report
        assert "for production" in report
        assert "RMSE Across All Runs" in report
        assert report_path.exists()


class TestConfig:
    """Tests for configuration."""
    
    def test_config_values(self):
        """Test that config has required values."""
        from pipeline.config import (
            DATASET_NAME,
            TEST_SIZE,
            DEFAULT_MODEL_TYPE,
            MODEL_CONFIGS,
        )
        
        assert DATASET_NAME == 'ml-100k'
        assert 0 < TEST_SIZE < 1
        assert DEFAULT_MODEL_TYPE in ['svd', 'nmf', 'knn']
        assert 'svd' in MODEL_CONFIGS


# =============================================================================
# Integration Tests
# =============================================================================
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test running the complete pipeline."""
        # This test is marked as slow and can be skipped in quick test runs
        # pytest tests/ -v -m "not slow"
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
