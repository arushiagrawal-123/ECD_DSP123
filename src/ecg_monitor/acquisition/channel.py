"""
Channel selection helpers for multi-channel WFDB records.

Keeps the channel-selection logic separate from the main loader so each
function has a single responsibility.
"""

from __future__ import annotations

def select_channel_index(
    sig_names: list[str],
    channel: str | int | None,
    preferred_channels: list[str] | None = None,
) -> int:
    """
    Resolve which channel index to use from a multi-channel record.

    Parameters
    ----------
    sig_names:
        List of channel names as returned by ``wfdb.rdrecord``.
    channel:
        Caller preference. Can be:
        - ``None`` — auto-select: use MLII if present, else the first channel.
        - ``int``  — direct index (validated against number of channels).
        - ``str``  — channel name (case-insensitive match required).
    preferred_channels:
        List of channel names in order of preference if auto-selecting.

    Returns
    -------
    int
        Zero-based index into *sig_names*.

    Raises
    ------
    ValueError
        If a named or indexed channel cannot be found.
    """
    if channel is None:
        if preferred_channels:
            for preferred in preferred_channels:
                try:
                    return _index_by_name(sig_names, preferred)
                except ValueError:
                    continue
        return 0  # fallback: first channel

    if isinstance(channel, int):
        if channel < 0 or channel >= len(sig_names):
            raise ValueError(
                f"Channel index {channel} is out of range for a record with "
                f"{len(sig_names)} channel(s): {sig_names}."
            )
        return channel

    # str: case-insensitive name match
    return _index_by_name(sig_names, channel)


def _index_by_name(sig_names: list[str], name: str) -> int:
    upper = name.upper()
    for i, s in enumerate(sig_names):
        if s.upper() == upper:
            return i
    raise ValueError(f"Channel '{name}' not found in record channels: {sig_names}.")
