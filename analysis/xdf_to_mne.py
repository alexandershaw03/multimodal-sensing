"""
Convert ATS multimodal XDF recording into MNE Raw FIF file. 
Converter extracts:
    ATS_EEG_RAW - Five-channel Emotiv Insight EEG.
and
    ATS_MARKERS - Manual experimental annotations.

EEG values are stored in the ATS LSL stream in microvolts : MNE expects EEG in volts.
Conversion, therefore, is applied before creating the RawArray.

Marker timestamps are preserved relative to the beginning of the recorded EEG stream (added as MNE annotations).

Usage
-----
python analysis/xdf_to_mne.py recording.xdf

Optional:
python analysis/xdf_to_mne.py recording.xdf \
    --output recording_raw.fif \
    --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pyxdf


# ============================================================================
# CONFIGURATION
# ============================================================================

EEG_STREAM_NAME = "ATS_EEG_RAW"
MARKER_STREAM_NAME = "ATS_MARKERS"

EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)


# ============================================================================
# XDF HELPERS
# ============================================================================


def stream_name(stream: dict) -> str:
    """Return LSL name stored in XDF stream."""

    return str(
        stream["info"]["name"][0]
    )


def build_stream_map(
    streams: list[dict],
) -> dict[str, dict]:
    """Index XDF streams by LSL stream name."""

    return {
        stream_name(stream): stream
        for stream in streams
    }


def require_stream(
    stream_map: dict[str, dict],
    name: str,
) -> dict:
    """Return required stream or raise useful error."""

    if name not in stream_map:
        available = ", ".join(
            sorted(stream_map)
        )

        raise RuntimeError(
            f"Required XDF stream '{name}' was not found.\n"
            f"Available streams: {available or 'none'}"
        )

    stream = stream_map[name]

    if len(
        stream.get(
            "time_stamps",
            [],
        )
    ) == 0:
        raise RuntimeError(
            f"XDF stream '{name}' contains no samples."
        )

    return stream


def decode_marker(value) -> str:
    """Convert XDF marker value into normal Python string."""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


# ============================================================================
# EEG
# ============================================================================


def get_sample_rates(
    eeg_stream: dict,
    timestamps: np.ndarray,
) -> tuple[float, float]:
    """
    Return nominal and timestamp-derived EEG sampling rates.

    LSL nominal rate used to construct the MNE Raw object.
    Timestamp-derived value is retained as a diagnostic.
    """

    if len(timestamps) < 2:
        raise RuntimeError(
            "At least two EEG timestamps are required."
        )

    deltas = np.diff(
        timestamps
    )

    if np.any(
        deltas <= 0
    ):
        raise RuntimeError(
            "EEG timestamps are not strictly increasing."
        )

    observed_rate = float(
        1.0
        /
        np.median(
            deltas
        )
    )

    try:
        nominal_rate = float(
            eeg_stream[
                "info"
            ][
                "nominal_srate"
            ][0]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        nominal_rate = 0.0

    if nominal_rate <= 0:
        nominal_rate = observed_rate

    return (
        nominal_rate,
        observed_rate,
    )


def create_mne_raw(
    eeg_stream: dict,
) -> tuple[
    mne.io.RawArray,
    np.ndarray,
    float,
]:
    """
    Convert ATS_EEG_RAW into MNE RawArray.

    Returns:
        raw
        original XDF EEG timestamps
        timestamp-derived sampling rate
    """

    eeg = np.asarray(
        eeg_stream[
            "time_series"
        ],
        dtype=float,
    )

    timestamps = np.asarray(
        eeg_stream[
            "time_stamps"
        ],
        dtype=float,
    )

    if eeg.ndim != 2:
        raise RuntimeError(
            f"Unexpected EEG array shape: {eeg.shape}"
        )

    if eeg.shape[0] != len(
        timestamps
    ):
        raise RuntimeError(
            "EEG sample count does not match timestamp count."
        )

    if eeg.shape[1] != len(
        EEG_CHANNELS
    ):
        raise RuntimeError(
            f"Expected {len(EEG_CHANNELS)} EEG channels "
            f"({', '.join(EEG_CHANNELS)}), "
            f"found {eeg.shape[1]}."
        )

    (
        nominal_rate,
        observed_rate,
    ) = get_sample_rates(
        eeg_stream,
        timestamps,
    )

    # ATS_EEG_RAW stores EEG in uV : MNE stores EEG in V.
    eeg_volts = (
        eeg
        /
        1_000_000.0
    ).T

    info = mne.create_info(
        ch_names=list(
            EEG_CHANNELS
        ),
        sfreq=nominal_rate,
        ch_types="eeg",
    )

    raw = mne.io.RawArray(
        eeg_volts,
        info,
        verbose=False,
    )

    montage = (
        mne.channels.make_standard_montage(
            "standard_1020"
        )
    )

    raw.set_montage(
        montage,
        on_missing="warn",
    )

    return (
        raw,
        timestamps,
        observed_rate,
    )


# ============================================================================
# MARKERS
# ============================================================================


def add_marker_annotations(
    raw: mne.io.RawArray,
    marker_stream: dict,
    eeg_timestamps: np.ndarray,
) -> int:
    """
    Add ATS_MARKERS to MNE Raw object as zero-duration annotation/s.

    Marker timestamps expressed relative to the first EEG timestamp.
    Markers outside EEG recording interval are ignored.
    """

    marker_timestamps = np.asarray(
        marker_stream[
            "time_stamps"
        ],
        dtype=float,
    )

    marker_values = marker_stream[
        "time_series"
    ]

    if len(marker_values) != len(
        marker_timestamps
    ):
        raise RuntimeError(
            "Marker sample count does not match marker timestamp count."
        )

    eeg_start = float(
        eeg_timestamps[0]
    )

    eeg_end = float(
        eeg_timestamps[-1]
    )

    annotation_onsets = []
    annotation_durations = []
    annotation_descriptions = []

    ignored = 0

    for marker, timestamp_raw in zip(
        marker_values,
        marker_timestamps,
    ):
        timestamp = float(
            timestamp_raw
        )

        if (
            timestamp < eeg_start
            or timestamp > eeg_end
        ):
            ignored += 1
            continue

        marker_value = (
            marker[0]
            if len(marker)
            else ""
        )

        description = decode_marker(
            marker_value
        )

        onset = (
            timestamp
            -
            eeg_start
        )

        annotation_onsets.append(
            onset
        )

        annotation_durations.append(
            0.0
        )

        annotation_descriptions.append(
            description
        )

    annotations = mne.Annotations(
        onset=annotation_onsets,
        duration=annotation_durations,
        description=annotation_descriptions,
    )

    raw.set_annotations(
        annotations
    )

    if ignored:
        print(
            f"Ignored {ignored} marker(s) outside "
            "the EEG recording interval."
        )

    return len(
        annotation_descriptions
    )


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse converter command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert ATS EEG and manual markers from XDF into an MNE FIF recording."
        )
    )

    parser.add_argument(
        "xdf",
        type=Path,
        help="Input XDF recording.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output FIF file. "
            "Defaults to <xdf_stem>_raw.fif beside the source XDF."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Open the MNE raw-data viewer after conversion."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


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

    output_file = (
        args.output
        .expanduser()
        .resolve()

        if args.output is not None

        else xdf_file.with_name(
            xdf_file.stem
            +
            "_raw.fif"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("========================================")
    print(" ATS XDF -> MNE")
    print("========================================")
    print()

    print(f"Loading: {xdf_file}")
    print()

    streams, _header = pyxdf.load_xdf(
        str(
            xdf_file
        )
    )

    stream_map = build_stream_map(
        streams
    )

    eeg_stream = require_stream(
        stream_map,
        EEG_STREAM_NAME,
    )

    marker_stream = require_stream(
        stream_map,
        MARKER_STREAM_NAME,
    )

    (
        raw,
        eeg_timestamps,
        observed_rate,
    ) = create_mne_raw(
        eeg_stream
    )

    annotation_count = add_marker_annotations(
        raw,
        marker_stream,
        eeg_timestamps,
    )

    print("EEG:")
    print(
        f"  Samples:          {raw.n_times}"
    )
    print(
        f"  Channels:         {len(raw.ch_names)}"
    )
    print(
        f"  Channel names:    {', '.join(raw.ch_names)}"
    )
    print(
        f"  Nominal rate:     {raw.info['sfreq']:.4f} Hz"
    )
    print(
        f"  Timestamp rate:   {observed_rate:.4f} Hz"
    )
    print(
        f"  Duration:         "
        f"{eeg_timestamps[-1] - eeg_timestamps[0]:.3f} s"
    )

    print()
    print(
        f"Annotations added: {annotation_count}"
    )

    if annotation_count:
        print()

        for annotation in raw.annotations:
            print(
                f"  {annotation['onset']:8.3f} s  "
                f"{annotation['description']}"
            )

    raw.save(
        output_file,
        overwrite=True,
        verbose=False,
    )

    print()
    print("========================================")
    print(" MNE OBJECT CREATED")
    print("========================================")
    print()
    print(raw)
    print()
    print(
        f"Saved:\n{output_file}"
    )
    print()

    if args.show:
        print(
            "Opening MNE viewer..."
        )

        raw.plot(
            duration=20,
            n_channels=len(
                EEG_CHANNELS
            ),
            scalings="auto",
            block=True,
        )


if __name__ == "__main__":
    main()
