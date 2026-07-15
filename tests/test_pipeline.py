"""Unit tests for pipeline.py components."""

import pytest
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Normalizer, OnlineStatistics, hours_to_index, CSVEventReader
from src.pipeline.normalizer import method_is_nonnegative


class TestNormalizer:
    """Tests for Normalizer class."""

    def test_method_is_nonnegative(self):
        """log/log1p methods are non-negative; zscore/minmax are sign-preserving."""
        assert method_is_nonnegative('log_zscore') is True
        assert method_is_nonnegative('log1p_zscore') is True
        assert method_is_nonnegative('zscore') is False
        assert method_is_nonnegative('minmax') is False

    def test_is_nonnegative_uses_variable_method(self):
        """is_nonnegative resolves the per-variable method (with default fallback)."""
        method_config = {
            'default': 'zscore',
            'methods': {'ap30': 'log1p_zscore', 'dst': 'zscore'},
        }
        normalizer = Normalizer(method_config=method_config)
        assert normalizer.is_nonnegative('ap30') is True    # log1p -> clip at 0
        assert normalizer.is_nonnegative('dst') is False     # signed -> keep sign
        assert normalizer.is_nonnegative('unlisted') is False  # falls back to default zscore

    def test_normalize_sdo_range(self):
        """SDO normalization should map [0, 255] to [-1, 1]."""
        normalizer = Normalizer()

        # Test boundary values
        data_min = np.array([0.0])
        data_max = np.array([255.0])
        data_mid = np.array([127.5])

        assert np.isclose(normalizer.normalize_sdo(data_min), -1.0)
        assert np.isclose(normalizer.normalize_sdo(data_max), 1.0)
        assert np.isclose(normalizer.normalize_sdo(data_mid), 0.0, atol=0.01)

    def test_normalize_sdo_shape_preserved(self):
        """SDO normalization should preserve input shape."""
        normalizer = Normalizer()

        data = np.random.randint(0, 256, size=(4, 3, 64, 64)).astype(np.float32)
        normalized = normalizer.normalize_sdo(data)

        assert normalized.shape == data.shape

    def test_normalize_omni_zscore(self):
        """OMNI normalization should apply z-score correctly."""
        stat_dict = {
            "test_var": {"mean": 10.0, "std": 2.0}
        }
        normalizer = Normalizer(stat_dict=stat_dict)

        data = np.array([10.0, 12.0, 8.0])
        normalized = normalizer.normalize_omni(data, "test_var")

        expected = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_denormalize_omni_inverse(self):
        """Denormalization should be inverse of normalization."""
        stat_dict = {
            "test_var": {"mean": 50.0, "std": 10.0}
        }
        normalizer = Normalizer(stat_dict=stat_dict)

        original = np.array([30.0, 50.0, 70.0, 100.0])
        normalized = normalizer.normalize_omni(original, "test_var")
        recovered = normalizer.denormalize_omni(normalized, "test_var")

        np.testing.assert_array_almost_equal(original, recovered)

    def test_normalize_omni_missing_variable(self):
        """Should raise KeyError for unknown variable."""
        stat_dict = {"known_var": {"mean": 0.0, "std": 1.0}}
        normalizer = Normalizer(stat_dict=stat_dict)

        with pytest.raises(KeyError):
            normalizer.normalize_omni(np.array([1.0]), "unknown_var")

    def test_denormalize_omni_missing_variable(self):
        """Should raise KeyError for unknown variable."""
        stat_dict = {"known_var": {"mean": 0.0, "std": 1.0}}
        normalizer = Normalizer(stat_dict=stat_dict)

        with pytest.raises(KeyError):
            normalizer.denormalize_omni(np.array([1.0]), "unknown_var")


class TestOnlineStatistics:
    """Tests for OnlineStatistics class."""

    def test_single_value(self):
        """Test with single value."""
        stats = OnlineStatistics()
        stats.update(np.array([5.0]))

        assert stats.mean == 5.0
        assert stats.std == 1.0  # Fallback for n < 2

    def test_multiple_values(self):
        """Test with multiple values."""
        stats = OnlineStatistics()
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        stats.update(data)

        assert stats.mean == 6.0
        assert np.isclose(stats.std, np.std(data), atol=0.01)

    def test_batch_updates(self):
        """Test incremental updates produce same result."""
        stats_batch = OnlineStatistics()
        stats_incremental = OnlineStatistics()

        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        all_data = np.concatenate([data1, data2])

        # Batch update
        stats_batch.update(all_data)

        # Incremental updates
        stats_incremental.update(data1)
        stats_incremental.update(data2)

        assert np.isclose(stats_batch.mean, stats_incremental.mean)
        assert np.isclose(stats_batch.std, stats_incremental.std)

    def test_ignores_nan_inf(self):
        """Should ignore NaN and Inf values."""
        stats = OnlineStatistics()
        data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0, -np.inf, 5.0])
        stats.update(data)

        # Only finite values: [1, 2, 3, 4, 5]
        assert stats.n == 5
        assert stats.mean == 3.0

    def test_get_stats_format(self):
        """get_stats should return dict with mean and std."""
        stats = OnlineStatistics()
        stats.update(np.array([1.0, 2.0, 3.0]))

        result = stats.get_stats()

        assert "mean" in result
        assert "std" in result
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)

    def test_multidimensional_input(self):
        """Should flatten and process multidimensional arrays."""
        stats = OnlineStatistics()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats.update(data)

        assert stats.n == 4
        assert stats.mean == 2.5


class TestHoursToIndex:
    """Tests for hours_to_index function."""

    def test_sdo_start_hours(self):
        """Test SDO start hours conversion."""
        # SDO: -96h from reference, 6h interval, base offset -168h
        # Index = (-96 - (-168)) / 6 = 72 / 6 = 12
        assert hours_to_index(-96, 6, -168) == 12

    def test_sdo_end_hours(self):
        """Test SDO end hours conversion."""
        # SDO: 0h (reference time), 6h interval, base offset -168h
        # Index = (0 - (-168)) / 6 = 168 / 6 = 28
        assert hours_to_index(0, 6, -168) == 28

    def test_omni_input_start(self):
        """Test OMNI input start hours conversion."""
        # OMNI input: -96h, 3h interval, base offset -168h
        # Index = (-96 - (-168)) / 3 = 72 / 3 = 24
        assert hours_to_index(-96, 3, -168) == 24

    def test_omni_input_end(self):
        """Test OMNI input end hours conversion."""
        # OMNI input: +72h, 3h interval, base offset -168h
        # Index = (72 - (-168)) / 3 = 240 / 3 = 80
        assert hours_to_index(72, 3, -168) == 80

    def test_omni_target_start(self):
        """Test OMNI target start hours conversion."""
        # OMNI target: +72h, 3h interval, base offset -168h
        # Index = (72 - (-168)) / 3 = 240 / 3 = 80
        assert hours_to_index(72, 3, -168) == 80

    def test_omni_target_end(self):
        """Test OMNI target end hours conversion."""
        # OMNI target: +144h, 3h interval, base offset -168h
        # Index = (144 - (-168)) / 3 = 312 / 3 = 104
        assert hours_to_index(144, 3, -168) == 104


class TestCSVEventReader:
    """Tests for CSVEventReader class."""

    def _create_sample_csv(self, tmpdir, rows=384, include_datetime=True):
        """Create a sample CSV event file for testing."""
        columns = [
            "v_avg", "v_min", "v_max",
            "np_avg", "np_min", "np_max",
            "t_avg", "t_min", "t_max",
            "bx_avg", "bx_min", "bx_max",
            "by_avg", "by_min", "by_max",
            "bz_avg", "bz_min", "bz_max",
            "bt_avg", "bt_min", "bt_max",
            "ap30", "hp30"
        ]

        import pandas as pd
        np.random.seed(42)
        data = {}
        if include_datetime:
            import datetime
            start = datetime.datetime(2021, 1, 5)
            data["datetime"] = [
                (start + datetime.timedelta(minutes=30 * i)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(rows)
            ]
        for col in columns:
            data[col] = np.random.rand(rows).astype(np.float32) * 100

        df = pd.DataFrame(data)
        path = os.path.join(str(tmpdir), "20210110000000.csv")
        df.to_csv(path, index=False)
        return path, columns

    def test_read_all_columns(self, tmp_path):
        """Should read all numeric columns, excluding datetime."""
        path, columns = self._create_sample_csv(tmp_path)
        data = CSVEventReader.read(path)
        assert data.shape == (384, 23)

    def test_read_selected_variables(self, tmp_path):
        """Should read only selected variables."""
        path, _ = self._create_sample_csv(tmp_path)
        selected = ["v_avg", "bz_avg", "ap30"]
        data = CSVEventReader.read(path, variables=selected)
        assert data.shape == (384, 3)

    def test_read_missing_variable(self, tmp_path):
        """Should raise KeyError for missing variable."""
        path, _ = self._create_sample_csv(tmp_path)
        with pytest.raises(KeyError):
            CSVEventReader.read(path, variables=["nonexistent_var"])

    def test_read_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CSVEventReader.read("/nonexistent/path.csv")

    def test_data_dtype(self, tmp_path):
        """Output should be float32."""
        path, _ = self._create_sample_csv(tmp_path)
        data = CSVEventReader.read(path)
        assert data.dtype == np.float32

    def test_custom_row_count(self, tmp_path):
        """Should handle different row counts."""
        path, _ = self._create_sample_csv(tmp_path, rows=48)
        data = CSVEventReader.read(path)
        assert data.shape[0] == 48


class TestNormalizerLogZscore:
    """Tests for log-based normalization methods (for new CSV variables)."""

    def test_log_zscore_roundtrip(self):
        """log_zscore normalize/denormalize should be inverse."""
        stat_dict = {
            "v_avg": {"log_mean": 5.5, "log_std": 0.5}
        }
        method_config = {"default": "zscore", "methods": {"v_avg": "log_zscore"}}
        normalizer = Normalizer(stat_dict=stat_dict, method_config=method_config)

        original = np.array([300.0, 500.0, 800.0])
        normalized = normalizer.normalize_omni(original, "v_avg")
        recovered = normalizer.denormalize_omni(normalized, "v_avg")

        np.testing.assert_array_almost_equal(original, recovered, decimal=3)

    def test_log1p_zscore_roundtrip(self):
        """log1p_zscore normalize/denormalize should be inverse."""
        stat_dict = {
            "ap30": {"log1p_mean": 2.0, "log1p_std": 1.0}
        }
        method_config = {"default": "zscore", "methods": {"ap30": "log1p_zscore"}}
        normalizer = Normalizer(stat_dict=stat_dict, method_config=method_config)

        original = np.array([0.0, 6.0, 50.0, 200.0])
        normalized = normalizer.normalize_omni(original, "ap30")
        recovered = normalizer.denormalize_omni(normalized, "ap30")

        np.testing.assert_array_almost_equal(original, recovered, decimal=3)

    def test_minmax_roundtrip(self):
        """minmax normalize/denormalize should be inverse."""
        stat_dict = {
            "hp30": {"min": 0.0, "max": 9.0}
        }
        method_config = {"default": "zscore", "methods": {"hp30": "minmax"}}
        normalizer = Normalizer(stat_dict=stat_dict, method_config=method_config)

        original = np.array([0.0, 4.5, 9.0])
        normalized = normalizer.normalize_omni(original, "hp30")
        recovered = normalizer.denormalize_omni(normalized, "hp30")

        np.testing.assert_array_almost_equal(original, recovered, decimal=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
