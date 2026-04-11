"""Tests for src/validators.py — MetricsAggregator and ResultsWriter components."""

import numpy as np
import pytest

from src.validators import MetricsAggregator, ResultsWriter


# ---------------------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------------------
class TestMetricsAggregator:
    """Tests for the MetricsAggregator class."""

    @staticmethod
    def _make_batch_result(batch_size=2, target_len=24, n_vars=1,
                           loss=0.5, mae=0.3, rmse=0.4, r2=0.8):
        """Create a synthetic batch result dictionary."""
        return {
            'loss': loss,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'targets': np.random.randn(batch_size, target_len, n_vars),
            'predictions': np.random.randn(batch_size, target_len, n_vars),
        }

    def test_initial_state(self):
        """Freshly created aggregator has empty collections."""
        agg = MetricsAggregator(target_variables=['ap30'])
        assert agg.losses == []
        assert agg.file_results == []

    def test_update_accumulates(self):
        """update() accumulates losses and file results."""
        agg = MetricsAggregator(target_variables=['ap30'])
        result = self._make_batch_result(batch_size=3)
        agg.update(result, file_names=['a.csv', 'b.csv', 'c.csv'])

        assert len(agg.losses) == 1
        assert agg.losses[0] == 0.5
        assert len(agg.file_results) == 3

    def test_update_with_cosine_sim(self):
        """update() stores cosine similarity when present."""
        agg = MetricsAggregator(target_variables=['ap30'])
        result = self._make_batch_result()
        result['cosine_sim'] = 0.95
        agg.update(result)

        assert len(agg.cosine_sims) == 1
        assert agg.cosine_sims[0] == pytest.approx(0.95)

    def test_update_without_file_names(self):
        """update() generates fallback names when file_names is None."""
        agg = MetricsAggregator(target_variables=['ap30'])
        result = self._make_batch_result(batch_size=2)
        agg.update(result, file_names=None)

        assert len(agg.file_results) == 2
        assert agg.file_results[0]['file_name'].startswith('sample_')

    def test_reset(self):
        """reset() clears all accumulated data."""
        agg = MetricsAggregator(target_variables=['ap30'])
        agg.update(self._make_batch_result())
        agg.reset()

        assert agg.losses == []
        assert agg.file_results == []
        assert agg.all_targets == []

    def test_get_summary_single_batch(self):
        """get_summary() returns correct structure for one batch."""
        agg = MetricsAggregator(target_variables=['ap30'])
        agg.update(self._make_batch_result(loss=0.5, mae=0.3, rmse=0.4, r2=0.8))

        summary = agg.get_summary()

        assert 'overall' in summary
        assert 'per_variable' in summary
        assert 'file_results' in summary
        assert 'total_samples' in summary

        overall = summary['overall']
        assert overall['average_loss'] == pytest.approx(0.5)
        assert overall['average_mae'] == pytest.approx(0.3)
        assert overall['average_rmse'] == pytest.approx(0.4)
        assert overall['average_r2'] == pytest.approx(0.8)

    def test_get_summary_multiple_batches(self):
        """get_summary() averages metrics across multiple batches."""
        agg = MetricsAggregator(target_variables=['ap30'])
        agg.update(self._make_batch_result(loss=1.0, mae=0.2, rmse=0.3, r2=0.9))
        agg.update(self._make_batch_result(loss=3.0, mae=0.4, rmse=0.5, r2=0.7))

        summary = agg.get_summary()
        overall = summary['overall']

        assert overall['average_loss'] == pytest.approx(2.0)
        assert overall['average_mae'] == pytest.approx(0.3)
        assert overall['std_loss'] == pytest.approx(1.0)

    def test_get_summary_no_cosine(self):
        """Cosine sim fields are None when no cosine data."""
        agg = MetricsAggregator(target_variables=['ap30'])
        agg.update(self._make_batch_result())

        summary = agg.get_summary()
        assert summary['overall']['average_cosine_sim'] is None

    def test_get_summary_empty_raises(self):
        """get_summary() raises ValueError when no data."""
        agg = MetricsAggregator(target_variables=['ap30'])
        with pytest.raises(ValueError, match="No data to summarize"):
            agg.get_summary()

    def test_per_variable_metrics(self):
        """Per-variable metrics are computed for each target variable."""
        agg = MetricsAggregator(target_variables=['ap30', 'hp30'])
        result = self._make_batch_result(n_vars=2)
        agg.update(result)

        summary = agg.get_summary()
        assert 'ap30' in summary['per_variable']
        assert 'hp30' in summary['per_variable']

        for var_metrics in summary['per_variable'].values():
            assert 'mae' in var_metrics
            assert 'rmse' in var_metrics
            assert 'r2_score' in var_metrics
            assert 'bias' in var_metrics


# ---------------------------------------------------------------------------
# ResultsWriter
# ---------------------------------------------------------------------------
class TestResultsWriter:
    """Tests for the ResultsWriter class."""

    @staticmethod
    def _make_results():
        """Create a synthetic results dictionary."""
        return {
            'overall': {
                'average_loss': 0.5,
                'std_loss': 0.1,
                'average_mae': 0.3,
                'std_mae': 0.05,
                'average_rmse': 0.4,
                'std_rmse': 0.06,
                'average_r2': 0.85,
                'std_r2': 0.02,
                'average_cosine_sim': None,
                'std_cosine_sim': None,
            },
            'per_variable': {
                'ap30': {
                    'mae': 0.3,
                    'rmse': 0.4,
                    'r2_score': 0.85,
                    'max_error': 1.2,
                    'median_absolute_error': 0.25,
                    'mape': 15.0,
                    'bias': -0.01,
                },
            },
            'total_samples': 10,
            'success_rate': 100.0,
            'file_results': [],
        }

    def test_write_summary_creates_file(self, tmp_path):
        """write_summary() creates a validation_results.txt file."""
        writer = ResultsWriter(str(tmp_path / "output"))
        writer.write_summary(self._make_results())

        summary_file = tmp_path / "output" / "validation_results.txt"
        assert summary_file.exists()

        content = summary_file.read_text()
        assert 'VALIDATION RESULTS SUMMARY' in content
        assert 'Average Loss' in content
        assert 'Average MAE' in content

    def test_write_summary_with_cosine(self, tmp_path):
        """write_summary() includes cosine sim when available."""
        results = self._make_results()
        results['overall']['average_cosine_sim'] = 0.92
        results['overall']['std_cosine_sim'] = 0.01

        writer = ResultsWriter(str(tmp_path / "output"))
        writer.write_summary(results)

        content = (tmp_path / "output" / "validation_results.txt").read_text()
        assert 'Cosine Sim' in content

    def test_write_csv_creates_file(self, tmp_path):
        """write_csv() creates a validation_results.csv file."""
        writer = ResultsWriter(str(tmp_path / "output"))
        file_results = [
            {
                'file_name': 'test_001.csv',
                'targets': np.array([[1.0, 2.0], [3.0, 4.0]]),
                'predictions': np.array([[1.1, 2.1], [3.1, 4.1]]),
            },
        ]

        writer.write_csv(file_results, target_variables=['ap30', 'hp30'])

        csv_file = tmp_path / "output" / "validation_results.csv"
        assert csv_file.exists()

        content = csv_file.read_text()
        assert 'file_name' in content
        assert 'target' in content
        assert 'prediction' in content

    def test_creates_output_dir(self, tmp_path):
        """ResultsWriter creates the output directory if it doesn't exist."""
        deep_path = tmp_path / "a" / "b" / "c"
        writer = ResultsWriter(str(deep_path))
        assert deep_path.exists()
