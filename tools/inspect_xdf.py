"""
Inspect contents of an XDF recording.
Used for providing a quick diagnostic summary of all streams in an XDF file, including: sample count; duration; nominal rate; timestamp-derived rate.

ATS EEG and marker streams receive additional content checks.

Usage
-----
python tools/inspect_xdf.py recording.xdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyxdf


EEG_STREAM_NAME = "ATS_EEG_RAW"
MARKER_STREAM_NAME = "ATS_MARKERS"

EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)


def decode_value(value) -> str:
    """Decode XDF string value."""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def get_stream_name(
    stream: dict,
) -> str:
    """Return XDF stream's LSL name."""

    return str(
        stream["info"]["name"][0]
    )


def get_stream_type(
    stream: dict,
) -> str:
    """Return XDF stream's LSL type."""

    return str(
        stream["info"]["type"][0]
    )


def nominal_rate(
    stream: dict,
) -> float:
    """Read stream's declared nominal sampling rate."""

    try:
        return float(
            stream["info"]["nominal_srate"][0]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return 0.0


def measured_rate(
    timestamps: np.ndarray,
) -> float | None:
    """Estimate rate from recorded timestamp sequence."""

    if len(timestamps) < 2:
        return None

    duration = float(
        timestamps[-1]
        -
        timestamps[0]
    )

    if duration <= 0:
        return None

    return float(
        (len(timestamps) - 1)
        /
        duration
    )


def print_stream_summary(
    stream: dict,
    index: int,
) -> None:
    """Print general information about one XDF stream."""

    name = get_stream_name(
        stream
    )

    stream_type = get_stream_type(
        stream
    )

    samples = np.asarray(
        stream["time_series"]
    )

    timestamps = np.asarray(
        stream["time_stamps"],
        dtype=float,
    )

    print("----------------------------------------")
    print(f"STREAM {index}")
    print("----------------------------------------")

    print(f"Name:          {name}")
    print(f"Type:          {stream_type}")
    print(f"Samples:       {len(samples)}")
    print(f"Data shape:    {samples.shape}")

    declared_rate = nominal_rate(
        stream
    )

    declared_text = (
        f"{declared_rate:.3f} Hz"
        if declared_rate > 0
        else "irregular"
    )

    print(
        f"Nominal rate:  {declared_text}"
    )

    if len(timestamps) > 0:
        duration = float(
            timestamps[-1]
            -
            timestamps[0]
        )

        print(
            f"Duration:      {duration:.3f} s"
        )

        rate = measured_rate(
            timestamps
        )

        if rate is not None:
            print(
                f"Measured rate: {rate:.3f} Hz"
            )

        print(
            f"First time:    {timestamps[0]:.6f}"
        )

        print(
            f"Last time:     {timestamps[-1]:.6f}"
        )

    print()


def inspect_eeg(
    stream: dict,
) -> None:
    """Display quick ATS_EEG content-check."""

    eeg = np.asarray(
        stream["time_series"],
        dtype=float,
    )

    print("ATS_EEG_RAW")
    print("-----------")

    print(
        f"Shape: {eeg.shape}"
    )

    if len(eeg) == 0:
        print("No EEG samples.")
        print()
        return

    if (
        eeg.ndim != 2
        or eeg.shape[1] != len(EEG_CHANNELS)
    ):
        print(
            "WARNING: EEG dimensions do not match the expected five-channel Insight layout."
        )

        print()
        return

    print("First sample:")

    for channel, value in zip(
        EEG_CHANNELS,
        eeg[0],
    ):
        print(
            f"  {channel:<3} = {value:.3f}"
        )

    print()

    print("Per-channel range:")

    for channel_index, channel in enumerate(
        EEG_CHANNELS
    ):
        values = eeg[
            :,
            channel_index
        ]

        print(
            f"  {channel:<3} "
            f"{np.nanmin(values):10.3f} "
            f"to "
            f"{np.nanmax(values):10.3f}"
        )

    print()


def inspect_markers(
    stream: dict,
) -> None:
    """Display all manual markers."""

    markers = stream[
        "time_series"
    ]

    timestamps = np.asarray(
        stream["time_stamps"],
        dtype=float,
    )

    print("ATS_MARKERS")
    print("-----------")

    if len(markers) == 0:
        print("No markers found.")
        print()
        return

    recording_reference = float(
        timestamps[0]
    )

    for sample, timestamp in zip(
        markers,
        timestamps,
    ):
        value = (
            sample[0]
            if len(sample)
            else ""
        )

        relative_time = (
            float(timestamp)
            -
            recording_reference
        )

        print(
            f"  +{relative_time:9.3f} s   "
            f"{decode_value(value)}"
        )

    print()


def parse_args() -> argparse.Namespace:
    """Parse XDF inspector arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect streams and contents inside XDF recording."
        )
    )

    parser.add_argument(
        "xdf",
        type=Path,
        help="XDF file to inspect.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xdf_file = (
        args.xdf
        .expanduser()
        .resolve()
    )

    if not xdf_file.is_file():
        raise FileNotFoundError(
            f"XDF file not found: {xdf_file}"
        )

    print()
    print("========================================")
    print(" ATS XDF INSPECTOR")
    print("========================================")
    print()

    print(
        f"Loading:\n{xdf_file}"
    )
    print()

    streams, _header = pyxdf.load_xdf(
        str(
            xdf_file
        )
    )

    print(
        f"Found {len(streams)} stream"
        f"{'' if len(streams) == 1 else 's'}."
    )

    print()

    for index, stream in enumerate(
        streams,
        start=1,
    ):
        print_stream_summary(
            stream,
            index,
        )

    print("========================================")
    print(" ATS CONTENT CHECKS")
    print("========================================")
    print()

    stream_map = {
        get_stream_name(stream): stream
        for stream in streams
    }

    if EEG_STREAM_NAME in stream_map:
        inspect_eeg(
            stream_map[
                EEG_STREAM_NAME
            ]
        )

    else:
        print(
            f"{EEG_STREAM_NAME}: not present.\n"
        )

    if MARKER_STREAM_NAME in stream_map:
        inspect_markers(
            stream_map[
                MARKER_STREAM_NAME
            ]
        )

    else:
        print(
            f"{MARKER_STREAM_NAME}: not present.\n"
        )

    print("========================================")
    print(" DONE")
    print("========================================")
    print()


if __name__ == "__main__":
    main()
