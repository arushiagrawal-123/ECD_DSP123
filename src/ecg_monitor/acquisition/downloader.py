"""
ECG record downloader — fetches MIT-BIH / PhysioNet files to a local cache directory.

Uses wfdb.dl_files to download only the three files needed for a single record:
the header (.hea), the signal data (.dat), and the annotation (.atr).
Download is skipped automatically if the files already exist and are the correct size.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import wfdb

# MIT-BIH Arrhythmia Database directory on PhysioNet
_MITDB_PN_DIR = "mitdb"

# File extensions required to read a record and its beat annotations
_REQUIRED_EXTENSIONS = (".hea", ".dat", ".atr")


def ensure_record_local(record_id: str, data_dir: Path) -> None:
    """
    Ensure a MIT-BIH record's files are present in *data_dir*.

    If any of the three required files (.hea, .dat, .atr) are missing,
    downloads them from PhysioNet. Already-present files are skipped.

    Parameters
    ----------
    record_id:
        MIT-BIH record name, e.g. ``"100"``.
    data_dir:
        Local directory where files are cached.

    Raises
    ------
    RuntimeError
        If the download fails (network error, record not found on PhysioNet, etc.).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    needed = [
        f"{record_id}{ext}"
        for ext in _REQUIRED_EXTENSIONS
        if not (data_dir / f"{record_id}{ext}").is_file()
    ]

    if not needed:
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wfdb.dl_files(
                db=_MITDB_PN_DIR,
                dl_dir=str(data_dir),
                files=needed,
                keep_subdirs=False,
                overwrite=False,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download record '{record_id}' from PhysioNet "
            f"(db='{_MITDB_PN_DIR}'). "
            f"Check your network connection and that the record ID is valid.\n"
            f"Original error: {exc}"
        ) from exc
