"""
Tests for the acquisition module.

Strategy
--------
Real wfdb calls go to PhysioNet over the network. To keep the test suite
fast, hermetic, and offline-capable we mock wfdb at the boundary and feed
synthetic data that mimics what a real MIT-BIH record looks like.

Tests that exercise the full download path (integration tests) are marked
``@pytest.mark.integration`` and skipped by default. Run them explicitly:
    pytest -m integration
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ecg_monitor.acquisition.channel import select_channel_index
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.core.models import ECGPackage, ReferenceAnnotations


# ---------------------------------------------------------------------------
# Fixtures — synthetic WFDB objects
# ---------------------------------------------------------------------------


def _make_mock_record(
    n_samples: int = 3600,
    n_channels: int = 2,
    fs: int = 360,
    sig_name: list[str] | None = None,
) -> MagicMock:
    """Return a mock that looks like a wfdb.Record with physical signal data."""
    rng = np.random.default_rng(0)
    p_signal = rng.standard_normal((n_samples, n_channels)).astype(np.float64)

    record = MagicMock()
    record.p_signal = p_signal
    record.fs = fs
    record.sig_name = sig_name or ["MLII", "V5"]
    return record


def _make_mock_annotation(
    n_beats: int = 30,
    n_samples: int = 3600,
) -> MagicMock:
    """Return a mock that looks like a wfdb.Annotation."""
    spacing = n_samples // (n_beats + 1)
    samples = np.array([spacing * (i + 1) for i in range(n_beats)])
    symbols = ["N"] * n_beats

    ann = MagicMock()
    ann.sample = samples
    ann.symbol = symbols
    return ann


# ---------------------------------------------------------------------------
# channel.select_channel_index
# ---------------------------------------------------------------------------


class TestSelectChannelIndex:
    def test_none_prefers_mlii(self) -> None:
        assert select_channel_index(["MLII", "V5"], None) == 0

    def test_none_falls_back_to_mli(self) -> None:
        assert select_channel_index(["MLI", "V1"], None) == 0

    def test_none_falls_back_to_first_when_no_preferred(self) -> None:
        assert select_channel_index(["ECG1", "ECG2"], None) == 0

    def test_int_index_valid(self) -> None:
        assert select_channel_index(["MLII", "V5"], 1) == 1

    def test_int_index_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            select_channel_index(["MLII", "V5"], 5)

    def test_str_name_match(self) -> None:
        assert select_channel_index(["MLII", "V5"], "V5") == 1

    def test_str_name_case_insensitive(self) -> None:
        assert select_channel_index(["mlii", "v5"], "MLII") == 0

    def test_str_name_not_found_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            select_channel_index(["MLII", "V5"], "I_DONT_EXIST")


# ---------------------------------------------------------------------------
# loader.load_record (mocked wfdb)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_wfdb(real_data_dir: Path):
    """Patch wfdb I/O to return synthetic data; skip actual network calls."""
    mock_record = _make_mock_record()
    mock_ann = _make_mock_annotation()

    with (
        patch(
            "ecg_monitor.acquisition.loader.ensure_record_local",
            return_value=None,
        ),
        patch(
            "ecg_monitor.acquisition.loader.wfdb.rdrecord",
            return_value=mock_record,
        ) as mock_rdr,
        patch(
            "ecg_monitor.acquisition.loader.wfdb.rdann",
            return_value=mock_ann,
        ) as mock_rda,
    ):
        yield real_data_dir, mock_rdr, mock_rda, mock_record, mock_ann


class TestLoadRecord:
    def test_returns_ecg_package(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert isinstance(pkg, ECGPackage)

    def test_record_id_preserved(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.record_id == "100"

    def test_sampling_rate_from_record(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.sampling_rate == 360

    def test_raw_signal_shape_is_1d(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.raw_signal.ndim == 1
        assert len(pkg.raw_signal) == 3600

    def test_raw_signal_dtype_float64(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.raw_signal.dtype == np.float64

    def test_filtered_signal_equals_raw_at_this_phase(
        self, mock_wfdb: tuple
    ) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        np.testing.assert_array_equal(pkg.raw_signal, pkg.filtered_signal)

    def test_filtered_signal_is_a_copy(self, mock_wfdb: tuple) -> None:
        """Mutating raw must not affect filtered and vice versa."""
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.raw_signal is not pkg.filtered_signal

    def test_quality_label_is_unset(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert pkg.quality.label == "UNSET"
        assert math.isnan(pkg.quality.snr_db)
        assert math.isnan(pkg.quality.rmse)
        assert math.isnan(pkg.quality.correlation)

    def test_annotations_are_non_empty(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        assert len(pkg.reference_annotations.sample_indices) > 0
        assert len(pkg.reference_annotations.symbols) > 0

    def test_annotations_lengths_match(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        ann = pkg.reference_annotations
        assert len(ann.sample_indices) == len(ann.symbols)

    def test_annotation_sample_indices_are_int(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        for idx in pkg.reference_annotations.sample_indices:
            assert isinstance(idx, int)

    def test_annotation_symbols_are_str(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        pkg = load_record("100", data_dir)
        for sym in pkg.reference_annotations.symbols:
            assert isinstance(sym, str)

    def test_channel_selection_by_name(self, mock_wfdb: tuple) -> None:
        data_dir, mock_rdr, _, mock_record, _ = mock_wfdb
        # Mock a record where MLII is index 0, V5 is index 1
        # Requesting V5 should extract column 1
        expected_col = mock_record.p_signal[:, 1]
        pkg = load_record("100", data_dir, channel="V5")
        np.testing.assert_array_almost_equal(pkg.raw_signal, expected_col)

    def test_channel_selection_by_index(self, mock_wfdb: tuple) -> None:
        data_dir, mock_rdr, _, mock_record, _ = mock_wfdb
        expected_col = mock_record.p_signal[:, 1]
        pkg = load_record("100", data_dir, channel=1)
        np.testing.assert_array_almost_equal(pkg.raw_signal, expected_col)

    def test_bad_channel_name_raises(self, mock_wfdb: tuple) -> None:
        data_dir, *_ = mock_wfdb
        with pytest.raises(ValueError, match="not found"):
            load_record("100", data_dir, channel="NONEXISTENT")

    def test_nan_samples_are_interpolated(self, real_data_dir: Path) -> None:
        mock_record = _make_mock_record()
        mock_record.p_signal[10:15, 0] = np.nan  # inject NaNs
        mock_ann = _make_mock_annotation()

        with (
            patch("ecg_monitor.acquisition.loader.ensure_record_local"),
            patch(
                "ecg_monitor.acquisition.loader.wfdb.rdrecord",
                return_value=mock_record,
            ),
            patch(
                "ecg_monitor.acquisition.loader.wfdb.rdann",
                return_value=mock_ann,
            ),
        ):
            pkg = load_record("100", real_data_dir)

        assert not np.any(np.isnan(pkg.raw_signal))

    def test_missing_record_raises_runtime_error(self, real_data_dir: Path) -> None:
        """A bad record ID that fails after download should raise RuntimeError."""
        with (
            patch("ecg_monitor.acquisition.loader.ensure_record_local"),
            patch(
                "ecg_monitor.acquisition.loader.wfdb.rdrecord",
                side_effect=FileNotFoundError("no such file"),
            ),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                load_record("BADRECORD", real_data_dir)

    def test_download_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        """A network error during download should raise RuntimeError with a clear message."""
        from ecg_monitor.acquisition.downloader import ensure_record_local

        with patch(
            "ecg_monitor.acquisition.downloader.wfdb.dl_files",
            side_effect=ConnectionError("timeout"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                ensure_record_local("BADRECORD", tmp_path)

    def test_empty_signal_raises_value_error(self, real_data_dir: Path) -> None:
        mock_record = MagicMock()
        mock_record.p_signal = np.empty((0, 2))
        mock_record.fs = 360
        mock_record.sig_name = ["MLII", "V5"]

        with (
            patch("ecg_monitor.acquisition.loader.ensure_record_local"),
            patch(
                "ecg_monitor.acquisition.loader.wfdb.rdrecord",
                return_value=mock_record,
            ),
        ):
            with pytest.raises(ValueError, match="no physical signal"):
                load_record("100", real_data_dir)


# ---------------------------------------------------------------------------
# Integration tests (skipped unless -m integration is passed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_load_real_record_100(real_data_dir: Path) -> None:
    """
    Downloads MIT-BIH record 100 from PhysioNet and verifies the full output.
    Requires network access.
    """
    pkg = load_record("100", real_data_dir)

    assert isinstance(pkg, ECGPackage)
    assert pkg.record_id == "100"
    assert pkg.sampling_rate == 360
    assert pkg.raw_signal.ndim == 1
    assert len(pkg.raw_signal) > 0
    assert len(pkg.reference_annotations.sample_indices) > 0
    assert all(isinstance(s, int) for s in pkg.reference_annotations.sample_indices)
    assert all(isinstance(s, str) for s in pkg.reference_annotations.symbols)
    assert pkg.quality.label == "UNSET"
