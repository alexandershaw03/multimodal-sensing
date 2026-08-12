"""
Validated EEG analysis: for ATS motor-response experiment.

Inputs
------
1. Multimodal XDF recording containing ATS_EEG_RAW.
2. Behavioural validation CSV from validation/validate_motor_trials.py.

Outputs
-------
- validated_trials_used.csv
- cue_aligned-epo.fif
- movement_aligned-epo.fif
- reaction-time, average EEG, PSD and ERD/ERS figures
- analysis_summary.txt

This analysis is exploratory. 
Behavioural validation confirms that the expected movement occurred. It is not a substitute for artefact rejection.

Usage
-----
python analysis/analyse_validated_motor_eeg.py recording.xdf

If validation CSV is omitted, the script looks beside the XDF for:
    <recording_stem>_validated_trials.csv

Optional:
python analysis/analyse_validated_motor_eeg.py recording.xdf validation.csv \
    --output-dir analysis_output --show
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import welch


# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

EEG_STREAM_NAME = "ATS_EEG_RAW"

EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)

EVENT_ID = {
    "LEFT": 1,
    "RIGHT": 2,
}


# Event-related EEG filtering
FILTER_LOW = 1.0
FILTER_HIGH = 40.0

# Mains power interference
NOTCH_FREQ = 50.0


# Epoch windows
CUE_TMIN = -2.0
CUE_TMAX = 3.0

MOVEMENT_TMIN = -2.0
MOVEMENT_TMAX = 2.0


# Baseline used for waveform and ERD/ERS calculation/s
BASELINE_START = -2.0
BASELINE_END = -1.0


# PSD range
PSD_LOW = 2.0
PSD_HIGH = 40.0


# Morlet TFR:
# 4-35 Hz, in 1 Hz steps.
TFR_FREQS = np.arange(
    4.0,
    36.0,
    1.0,
)

# freq / 2 gives approx 0.5s wavelet duration.
TFR_N_CYCLES = (
    TFR_FREQS
    / 2.0
)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class EEGRecording:
    """
    EEG samples and timing extracted from XDF.
    """

    data_microvolts: np.ndarray
    timestamps: np.ndarray

    sfreq: float
    observed_sfreq: float


@dataclass(frozen=True)
class EpochBuildResult:
    """
    MNE epochs and event-to-nearest-sample timing diagnostics.
    """

    epochs: mne.Epochs
    timing_errors_ms: np.ndarray


# ============================================================================
# INPUT / STREAM HELPERS
# ============================================================================


def stream_name(
    stream: dict,
) -> str:
    """
    Return XDF stream's LSL name.
    """

    return str(
        stream[
            "info"
        ][
            "name"
        ][0]
    )


def build_stream_map(
    streams: list[dict],
) -> dict[str, dict]:
    """
    Index XDF streams by LSL name.
    """

    return {
        stream_name(stream): stream
        for stream in streams
    }


def require_stream(
    stream_map: dict[str, dict],
    name: str,
) -> dict:
    """
    Return required XDF stream or raise useful error.
    """

    if name not in stream_map:

        available = ", ".join(
            sorted(
                stream_map
            )
        )

        raise RuntimeError(
            f"Required XDF stream '{name}' was not found.\n"
            f"Available streams: {available or 'none'}"
        )

    stream = stream_map[
        name
    ]

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


def load_eeg_recording(
    xdf_file: Path,
) -> EEGRecording:
    """
    Load ATS_EEG_RAW and validate basic structure.
    """

    streams, _header = pyxdf.load_xdf(
        str(
            xdf_file
        )
    )

    eeg_stream = require_stream(
        build_stream_map(
            streams
        ),
        EEG_STREAM_NAME,
    )

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

    if len(
        timestamps
    ) < 2:

        raise RuntimeError(
            "At least two EEG timestamps are required."
        )

    timestamp_deltas = np.diff(
        timestamps
    )

    if np.any(
        timestamp_deltas
        <= 0
    ):

        raise RuntimeError(
            "EEG timestamps are not strictly increasing."
        )

    observed_sfreq = float(
        1.0
        /
        np.median(
            timestamp_deltas
        )
    )

    nominal_sfreq = float(
        eeg_stream[
            "info"
        ][
            "nominal_srate"
        ][0]
    )

    sfreq = (
        nominal_sfreq
        if nominal_sfreq > 0
        else observed_sfreq
    )

    return EEGRecording(
        data_microvolts=eeg,
        timestamps=timestamps,
        sfreq=sfreq,
        observed_sfreq=observed_sfreq,
    )


def parse_valid_boolean(
    series: pd.Series,
) -> pd.Series:
    """
    Interpret common representations of "True".
    """

    return (
        series
        .astype(
            str
        )
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def load_validated_trials(
    validation_csv: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load validation output and return full table plus valid subset.
    """

    validation = pd.read_csv(
        validation_csv
    )

    required_columns = {
        "trial",
        "side",
        "overall_valid",
        "reaction_time",
        "cue_time",
        "movement_time",
        "hold_time",
        "rest_time",
    }

    missing = sorted(
        required_columns
        -
        set(
            validation.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Validation CSV is missing required columns: "
            +
            ", ".join(
                missing
            )
        )

    validation = validation.copy()

    validation[
        "valid_bool"
    ] = parse_valid_boolean(
        validation[
            "overall_valid"
        ]
    )

    valid_trials = (
        validation[
            validation[
                "valid_bool"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        valid_trials
    ) == 0:

        raise RuntimeError(
            "No behaviourally valid trials are available for EEG analysis."
        )

    valid_trials[
        "side"
    ] = (
        valid_trials[
            "side"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    invalid_sides = sorted(
        set(
            valid_trials[
                "side"
            ]
        )
        -
        set(
            EVENT_ID
        )
    )

    if invalid_sides:

        raise RuntimeError(
            "Validated trials contain unsupported side labels: "
            +
            ", ".join(
                invalid_sides
            )
        )

    numeric_columns = (
        "trial",
        "reaction_time",
        "cue_time",
        "movement_time",
        "hold_time",
        "rest_time",
    )

    for column in numeric_columns:

        valid_trials[
            column
        ] = pd.to_numeric(
            valid_trials[
                column
            ],
            errors="coerce",
        )

    critical_columns = [
        "trial",
        "reaction_time",
        "cue_time",
        "movement_time",
    ]

    if (
        valid_trials[
            critical_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "One or more valid trials contain missing or non-numeric trial, reaction_time, cue_time, or movement_time values."
        )

    return (
        validation,
        valid_trials,
    )


# ============================================================================
# MNE PREPARATION
# ============================================================================


def create_filtered_raw(
    recording: EEGRecording,
) -> mne.io.RawArray:
    """
    Create MNE RawArray and apply analysis filters.
    """

    # ATS_EEG_RAW stores microvolts; MNE expects volts.
    eeg_volts = (
        recording.data_microvolts
        /
        1_000_000.0
    ).T

    info = mne.create_info(
        ch_names=list(
            EEG_CHANNELS
        ),
        sfreq=recording.sfreq,
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

    filtered = raw.copy()

    filtered.notch_filter(
        freqs=[
            NOTCH_FREQ
        ],
        picks="eeg",
        verbose=False,
    )

    filtered.filter(
        l_freq=FILTER_LOW,
        h_freq=FILTER_HIGH,
        picks="eeg",
        verbose=False,
    )

    return filtered


# ============================================================================
# XDF EVENT -> EEG SAMPLE ALIGNMENT
# ============================================================================


def nearest_eeg_sample(
    eeg_timestamps: np.ndarray,
    event_time: float,
) -> tuple[
    int | None,
    float | None,
]:
    """
    Find recorded EEG sample nearest to XDF event timestamp.

    Deliberately uses the actual XDF timestamp sequence rather than:

        sample_number = elapsed_time * nominal_sample_rate

    Timing error returned:

        nearest EEG timestamp - event timestamp
    """

    insertion = int(
        np.searchsorted(
            eeg_timestamps,
            event_time,
        )
    )

    candidates: list[int] = []

    if insertion < len(
        eeg_timestamps
    ):

        candidates.append(
            insertion
        )

    if insertion > 0:

        candidates.append(
            insertion - 1
        )

    if not candidates:

        return (
            None,
            None,
        )

    best_index = min(
        candidates,
        key=lambda index: abs(
            eeg_timestamps[
                index
            ]
            -
            event_time
        ),
    )

    timing_error = float(
        eeg_timestamps[
            best_index
        ]
        -
        event_time
    )

    return (
        best_index,
        timing_error,
    )


def build_epochs(
    *,
    raw_filtered: mne.io.RawArray,
    eeg_timestamps: np.ndarray,
    valid_trials: pd.DataFrame,
    alignment_name: str,
    timestamp_column: str,
    tmin: float,
    tmax: float,
) -> EpochBuildResult:
    """
    Build MNE epochs from one event timestamp column.
    """

    events: list[
        list[int]
    ] = []

    metadata_rows: list[
        dict[str, object]
    ] = []

    timing_errors: list[
        float
    ] = []

    print()

    print(
        f"Building "
        f"{alignment_name.lower()}-aligned epochs..."
    )

    for _, row in valid_trials.iterrows():

        trial_number = int(
            row[
                "trial"
            ]
        )

        side = str(
            row[
                "side"
            ]
        )

        event_time = float(
            row[
                timestamp_column
            ]
        )

        if not math.isfinite(
            event_time
        ):

            print(
                f"Skipping trial {trial_number}: "
                f"{timestamp_column} is not finite."
            )

            continue

        if (
            event_time
            + tmin
            <
            eeg_timestamps[0]
        ):

            print(
                f"Skipping trial {trial_number}: "
                "epoch extends before EEG recording start."
            )

            continue

        if (
            event_time
            + tmax
            >
            eeg_timestamps[-1]
        ):

            print(
                f"Skipping trial {trial_number}: "
                "epoch extends beyond EEG recording end."
            )

            continue

        (
            sample_index,
            timing_error,
        ) = nearest_eeg_sample(
            eeg_timestamps,
            event_time,
        )

        if (
            sample_index is None
            or timing_error is None
        ):

            continue

        timing_errors.append(
            timing_error
        )

        events.append(
            [
                int(
                    sample_index
                ),
                0,
                EVENT_ID[
                    side
                ],
            ]
        )

        metadata_rows.append(
            {
                "trial": trial_number,
                "side": side,

                "reaction_time":
                    float(
                        row[
                            "reaction_time"
                        ]
                    ),

                "hold_time":
                    float(
                        row[
                            "hold_time"
                        ]
                    ),

                "rest_time":
                    float(
                        row[
                            "rest_time"
                        ]
                    ),

                "alignment":
                    alignment_name,

                "event_timestamp":
                    event_time,

                "sample_index":
                    int(
                        sample_index
                    ),

                "sample_timing_error_ms":
                    float(
                        timing_error
                        * 1000.0
                    ),
            }
        )

    if not events:

        raise RuntimeError(
            f"No usable "
            f"{alignment_name.lower()} events were available."
        )

    events_array = np.asarray(
        events,
        dtype=int,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    # Only include event IDs actually present in this epoch set.
    # (To make analysis more robust to small, test, datasets; where one condition may be missing)
    present_codes = set(
        events_array[
            :,
            2
        ]
    )

    present_event_id = {
        side: code

        for side, code
        in EVENT_ID.items()

        if code in present_codes
    }

    epochs = mne.Epochs(
        raw_filtered,
        events_array,

        event_id=present_event_id,

        tmin=tmin,
        tmax=tmax,

        baseline=None,

        preload=True,

        metadata=metadata,

        reject_by_annotation=True,

        verbose=False,
    )

    timing_errors_ms = (
        np.asarray(
            timing_errors,
            dtype=float,
        )
        *
        1000.0
    )

    print(
        f"{len(epochs)} epochs created."
    )

    print(
        "Mean absolute event/sample mismatch: "
        f"{np.mean(np.abs(timing_errors_ms)):.3f} ms"
    )

    print(
        "Worst absolute event/sample mismatch: "
        f"{np.max(np.abs(timing_errors_ms)):.3f} ms"
    )

    return EpochBuildResult(
        epochs=epochs,
        timing_errors_ms=timing_errors_ms,
    )


# ============================================================================
# PLOTTING HELPERS
# ============================================================================


def save_figure(
    fig: plt.Figure,
    path: Path,
    show: bool,
) -> None:
    """
    Save a figure and close it, unless interactive display requested.
    """

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    if not show:

        plt.close(
            fig
        )


def plot_reaction_times(
    valid_trials: pd.DataFrame,
    output_file: Path,
    show: bool,
) -> None:
    """
    Plot cue-to-movement reaction times, for valid trials.
    """

    trial_numbers = (
        valid_trials[
            "trial"
        ]
        .to_numpy(
            dtype=int
        )
    )

    reaction_times = (
        valid_trials[
            "reaction_time"
        ]
        .to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    ax.scatter(
        trial_numbers,
        reaction_times,
        s=80,
    )

    for (
        trial,
        side,
        reaction,
    ) in zip(
        trial_numbers,
        valid_trials[
            "side"
        ],
        reaction_times,
    ):

        ax.text(
            trial,
            reaction + 0.025,
            side,
            ha="center",
            fontsize=9,
        )

    reaction_mean = float(
        np.mean(
            reaction_times
        )
    )

    ax.axhline(
        reaction_mean,
        linestyle="--",
        alpha=0.7,
        label=(
            f"Mean = "
            f"{reaction_mean:.3f} s"
        ),
    )

    ax.set_title(
        "ATS Motor Experiment — Valid Trial Reaction Times"
    )

    ax.set_xlabel(
        "Trial"
    )

    ax.set_ylabel(
        "Cue → movement onset (s)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output_file,
        show,
    )


def baseline_corrected_microvolts(
    epochs: mne.Epochs,
) -> np.ndarray:
    """
    Apply configured baseline and return EEG in mV
    """

    corrected = (
        epochs.copy()
    )

    corrected.apply_baseline(
        (
            BASELINE_START,
            BASELINE_END,
        )
    )

    return (
        corrected.get_data()
        *
        1_000_000.0
    )


def plot_condition_averages(
    *,
    epochs: mne.Epochs,
    title: str,
    output_file: Path,
    zero_label: str,
    show: bool,
) -> None:
    """
    Plot LEFT / RIGHT, baseline-corrected, average EEG.
    """

    times = epochs.times

    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,

        figsize=(
            13,
            12,
        ),

        sharex=True,
    )

    for channel_index, (
        channel,
        ax,
    ) in enumerate(
        zip(
            EEG_CHANNELS,
            axes,
        )
    ):

        plotted = False

        for side in (
            "LEFT",
            "RIGHT",
        ):

            if side not in epochs.event_id:

                continue

            side_epochs = epochs[
                side
            ]

            if len(
                side_epochs
            ) == 0:

                continue

            data = (
                baseline_corrected_microvolts(
                    side_epochs
                )
            )

            average = np.mean(
                data[
                    :,
                    channel_index,
                    :,
                ],
                axis=0,
            )

            ax.plot(
                times,
                average,
                label=(
                    f"{side} "
                    f"(n={len(side_epochs)})"
                ),
            )

            plotted = True

        ax.axvline(
            0.0,
            linestyle="--",
            alpha=0.7,
        )

        ax.axhline(
            0.0,
            linewidth=0.7,
            alpha=0.4,
        )

        ax.set_ylabel(
            f"{channel}\nµV"
        )

        ax.grid(
            alpha=0.20
        )

        if plotted:

            ax.legend(
                loc="upper right"
            )

    axes[-1].set_xlabel(
        f"Time relative to {zero_label} (s)"
    )

    fig.suptitle(
        title,
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    save_figure(
        fig,
        output_file,
        show,
    )


# ============================================================================
# POWER SPECTRAL DENSITY
# ============================================================================


def calculate_psd(
    epochs: mne.Epochs,
    sfreq: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate Welch PSD for every epoch and EEG channel.
    """

    data = (
        epochs.get_data()
    )

    nperseg = min(
        256,
        data.shape[-1],
    )

    freqs, power = welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        axis=-1,
    )

    # V² / Hz -> µV² / Hz
    power = (
        power
        *
        1e12
    )

    keep = (
        (freqs >= PSD_LOW)
        &
        (freqs <= PSD_HIGH)
    )

    return (
        freqs[
            keep
        ],

        power[
            ...,
            keep
        ],
    )


def plot_movement_psd(
    *,
    movement_epochs: mne.Epochs,
    sfreq: float,
    output_file: Path,
    show: bool,
) -> None:
    """
    Plot movement-aligned LEFT / RIGHT EEG PSD.
    """

    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,

        figsize=(
            12,
            12,
        ),

        sharex=True,
    )

    plotted = False

    for side in (
        "LEFT",
        "RIGHT",
    ):

        if side not in movement_epochs.event_id:

            continue

        side_epochs = movement_epochs[
            side
        ]

        if len(
            side_epochs
        ) == 0:

            continue

        (
            freqs,
            power,
        ) = calculate_psd(
            side_epochs,
            sfreq,
        )

        mean_power = np.mean(
            power,
            axis=0,
        )

        db_power = (
            10.0
            *
            np.log10(
                mean_power
                +
                1e-20
            )
        )

        for channel_index, ax in enumerate(
            axes
        ):

            ax.plot(
                freqs,

                db_power[
                    channel_index
                ],

                label=(
                    f"{side} "
                    f"(n={len(side_epochs)})"
                ),
            )

        plotted = True

    for channel, ax in zip(
        EEG_CHANNELS,
        axes,
    ):

        ax.set_ylabel(
            f"{channel}\ndB"
        )

        ax.grid(
            alpha=0.20
        )

        if plotted:

            ax.legend(
                loc="upper right"
            )

    axes[-1].set_xlabel(
        "Frequency (Hz)"
    )

    fig.suptitle(
        "Movement-Aligned EEG-Power Spectral Density",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    save_figure(
        fig,
        output_file,
        show,
    )


# ============================================================================
# TIME-FREQUENCY / ERD-ERS
# ============================================================================


def calculate_erds(
    epochs: mne.Epochs,
    sfreq: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate trial-wise baseline-normalised Morlet power.

    Output:
        0%   = baseline power
        <0%  = reduced power / desynchronisation
        >0%  = increased power / synchronisation
    """

    data = (
        epochs.get_data()
    )

    power = (
        mne.time_frequency.tfr_array_morlet(
            data,

            sfreq=sfreq,

            freqs=TFR_FREQS,

            n_cycles=TFR_N_CYCLES,

            output="power",

            zero_mean=True,

            n_jobs=1,

            verbose=False,
        )
    )

    times = epochs.times

    baseline_mask = (
        (times >= BASELINE_START)
        &
        (times <= BASELINE_END)
    )

    if not np.any(
        baseline_mask
    ):

        raise RuntimeError(
            "Configured ERD/ERS baseline is outside the epoch window."
        )

    baseline_power = np.mean(
        power[
            ...,
            baseline_mask,
        ],

        axis=-1,

        keepdims=True,
    )

    # Avoid divide-by-zero instability.
    baseline_power = np.maximum(
        baseline_power,
        1e-30,
    )

    erds = (
        (
            power
            -
            baseline_power
        )
        /
        baseline_power
        *
        100.0
    )

    average_erds = np.mean(
        erds,
        axis=0,
    )

    return (
        times,
        average_erds,
    )


def plot_tfr_condition(
    *,
    epochs: mne.Epochs,
    sfreq: float,
    side: str,
    alignment_name: str,
    zero_label: str,
    output_file: Path,
    show: bool,
) -> None:
    """
    Plot one condition's channel-wise ERD/ERS maps.
    """

    if side not in epochs.event_id:

        print(
            f"Skipping {alignment_name} {side} TFR: "
            "no epochs for this condition."
        )

        return

    side_epochs = epochs[
        side
    ]

    if len(
        side_epochs
    ) == 0:

        return

    print(
        f"Calculating "
        f"{alignment_name} {side} TFR "
        f"(n={len(side_epochs)})..."
    )

    (
        times,
        erds,
    ) = calculate_erds(
        side_epochs,
        sfreq,
    )

    # Shared robust colour scale across all channels within this figure.
    robust_limit = float(
        np.nanpercentile(
            np.abs(
                erds
            ),
            97,
        )
    )

    robust_limit = max(
        robust_limit,
        1.0,
    )

    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,

        figsize=(
            13,
            15,
        ),

        sharex=True,
    )

    image = None

    for channel_index, (
        channel,
        ax,
    ) in enumerate(
        zip(
            EEG_CHANNELS,
            axes,
        )
    ):

        image = ax.imshow(
            erds[
                channel_index
            ],

            aspect="auto",

            origin="lower",

            extent=[
                times[0],
                times[-1],
                TFR_FREQS[0],
                TFR_FREQS[-1],
            ],

            cmap="RdBu_r",

            vmin=-robust_limit,
            vmax=robust_limit,
        )

        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_ylabel(
            f"{channel}\nHz"
        )

    axes[-1].set_xlabel(
        f"Time relative to {zero_label} (s)"
    )

    fig.suptitle(
        f"{alignment_name}-Aligned {side} ERD/ERS "
        f"— n={len(side_epochs)}",

        fontsize=16,
    )

    if image is not None:

        colorbar = fig.colorbar(
            image,
            ax=axes,
            pad=0.015,
        )

        colorbar.set_label(
            "Power change from baseline (%)"
        )

    fig.subplots_adjust(
        left=0.10,
        right=0.88,
        bottom=0.06,
        top=0.94,
        hspace=0.18,
    )

    save_figure(
        fig,
        output_file,
        show,
    )


# ============================================================================
# SUMMARY
# ============================================================================


def write_summary(
    *,
    output_file: Path,
    xdf_file: Path,
    validation_csv: Path,
    validation: pd.DataFrame,
    valid_trials: pd.DataFrame,
    recording: EEGRecording,
    cue_timing_errors_ms: np.ndarray,
    movement_timing_errors_ms: np.ndarray,
) -> None:
    """
    Write concise summary without exposing machine-specific paths.
    """

    reaction_times = (
        valid_trials[
            "reaction_time"
        ]
        .to_numpy(
            dtype=float
        )
    )

    left_count = int(
        np.sum(
            valid_trials[
                "side"
            ]
            ==
            "LEFT"
        )
    )

    right_count = int(
        np.sum(
            valid_trials[
                "side"
            ]
            ==
            "RIGHT"
        )
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ATS VALIDATED MOTOR EEG ANALYSIS\n"
        )

        file.write(
            "================================\n\n"
        )

        # Filenames only (safe to publish without exposing local paths):
        file.write(
            f"XDF source: {xdf_file.name}\n"
        )

        file.write(
            f"Validation source: "
            f"{validation_csv.name}\n\n"
        )

        file.write(
            f"Total commanded trials: "
            f"{len(validation)}\n"
        )

        file.write(
            f"Valid trials: "
            f"{len(valid_trials)}\n"
        )

        file.write(
            f"Valid LEFT: "
            f"{left_count}\n"
        )

        file.write(
            f"Valid RIGHT: "
            f"{right_count}\n"
        )

        file.write(
            "\nReaction time mean: "
            f"{np.mean(reaction_times):.4f} s\n"
        )

        file.write(
            "Reaction time SD: "
            f"{np.std(reaction_times):.4f} s\n"
        )

        file.write(
            "Reaction time min: "
            f"{np.min(reaction_times):.4f} s\n"
        )

        file.write(
            "Reaction time max: "
            f"{np.max(reaction_times):.4f} s\n"
        )

        file.write(
            "\nEEG nominal sample rate: "
            f"{recording.sfreq:.4f} Hz\n"
        )

        file.write(
            "EEG median timestamp-derived rate: "
            f"{recording.observed_sfreq:.4f} Hz\n"
        )

        file.write(
            "\nEEG filtering: "
            f"{FILTER_LOW:.1f}-{FILTER_HIGH:.1f} Hz\n"
        )

        file.write(
            f"Notch: "
            f"{NOTCH_FREQ:.1f} Hz\n"
        )

        file.write(
            "\nCue event/sample mean absolute mismatch: "
            f"{np.mean(np.abs(cue_timing_errors_ms)):.3f} ms\n"
        )

        file.write(
            "Cue event/sample worst absolute mismatch: "
            f"{np.max(np.abs(cue_timing_errors_ms)):.3f} ms\n"
        )

        file.write(
            "Movement event/sample mean absolute mismatch: "
            f"{np.mean(np.abs(movement_timing_errors_ms)):.3f} ms\n"
        )

        file.write(
            "Movement event/sample worst absolute mismatch: "
            f"{np.max(np.abs(movement_timing_errors_ms)):.3f} ms\n"
        )

        file.write(
            "\nIMPORTANT:\n"
            "This analysis is exploratory. "
            "Behavioural validation does not constitute "
            "EEG artefact rejection.\n"
        )


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse analysis command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Analyse behaviourally validated "
            "ATS motor-response EEG trials."
        )
    )

    parser.add_argument(
        "xdf",
        type=Path,
        help=(
            "Multimodal XDF recording."
        ),
    )

    parser.add_argument(
        "validation_csv",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Validation CSV. "
            "If omitted, defaults to "
            "<xdf_stem>_validated_trials.csv "
            "beside the XDF."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Analysis output directory. "
            "Defaults to <xdf_stem>_analysis "
            "beside the XDF."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display saved figures "
            "interactively at the end."
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
            f"XDF file not found: "
            f"{xdf_file}"
        )

    validation_csv = (
        args.validation_csv
        .expanduser()
        .resolve()

        if args.validation_csv is not None

        else xdf_file.with_name(
            xdf_file.stem
            +
            "_validated_trials.csv"
        )
    )

    if not validation_csv.is_file():

        raise FileNotFoundError(
            f"Validation CSV not found: "
            f"{validation_csv}\n"
            "Run validation/validate_motor_trials.py "
            "first or provide the validation CSV explicitly."
        )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()

        if args.output_dir is not None

        else xdf_file.with_name(
            xdf_file.stem
            +
            "_analysis"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "========================================"
    )
    print(
        " ATS VALIDATED MOTOR EEG ANALYSIS"
    )
    print(
        "========================================"
    )
    print()

    print(
        f"XDF:        {xdf_file}"
    )

    print(
        f"Validation: {validation_csv}"
    )

    print(
        f"Output:     {output_dir}"
    )

    print()

    # ----------------------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------------------

    print(
        "Loading EEG from XDF..."
    )

    recording = load_eeg_recording(
        xdf_file
    )

    print(
        f"EEG samples:             "
        f"{recording.data_microvolts.shape[0]}"
    )

    print(
        f"EEG channels:            "
        f"{recording.data_microvolts.shape[1]}"
    )

    print(
        f"Nominal analysis rate:   "
        f"{recording.sfreq:.4f} Hz"
    )

    print(
        f"Median timestamp rate:   "
        f"{recording.observed_sfreq:.4f} Hz"
    )

    print()

    (
        validation,
        valid_trials,
    ) = load_validated_trials(
        validation_csv
    )

    left_count = int(
        np.sum(
            valid_trials[
                "side"
            ]
            ==
            "LEFT"
        )
    )

    right_count = int(
        np.sum(
            valid_trials[
                "side"
            ]
            ==
            "RIGHT"
        )
    )

    print(
        f"Total commanded trials:   "
        f"{len(validation)}"
    )

    print(
        f"Valid behavioural trials: "
        f"{len(valid_trials)}"
    )

    print(
        f"Valid LEFT:               "
        f"{left_count}"
    )

    print(
        f"Valid RIGHT:              "
        f"{right_count}"
    )

    valid_trials.to_csv(
        output_dir
        /
        "validated_trials_used.csv",

        index=False,
    )

    # ----------------------------------------------------------------------
    # Filter and epoch
    # ----------------------------------------------------------------------

    print()

    print(
        f"Filtering EEG: "
        f"{FILTER_LOW:.1f}-{FILTER_HIGH:.1f} Hz, "
        f"{NOTCH_FREQ:.1f} Hz notch"
    )

    raw_filtered = create_filtered_raw(
        recording
    )

    cue_result = build_epochs(
        raw_filtered=raw_filtered,

        eeg_timestamps=
            recording.timestamps,

        valid_trials=
            valid_trials,

        alignment_name=
            "CUE",

        timestamp_column=
            "cue_time",

        tmin=
            CUE_TMIN,

        tmax=
            CUE_TMAX,
    )

    movement_result = build_epochs(
        raw_filtered=raw_filtered,

        eeg_timestamps=
            recording.timestamps,

        valid_trials=
            valid_trials,

        alignment_name=
            "MOVEMENT",

        timestamp_column=
            "movement_time",

        tmin=
            MOVEMENT_TMIN,

        tmax=
            MOVEMENT_TMAX,
    )

    cue_epochs = (
        cue_result.epochs
    )

    movement_epochs = (
        movement_result.epochs
    )

    cue_epochs.save(
        output_dir
        /
        "cue_aligned-epo.fif",

        overwrite=True,
    )

    movement_epochs.save(
        output_dir
        /
        "movement_aligned-epo.fif",

        overwrite=True,
    )

    # ----------------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------------

    print()
    print(
        "Generating figures..."
    )

    plot_reaction_times(
        valid_trials,

        output_dir
        /
        "01_reaction_times.png",

        args.show,
    )

    plot_condition_averages(
        epochs=
            cue_epochs,

        title=
            "Cue-Aligned EEG — LEFT vs RIGHT",

        output_file=
            output_dir
            /
            "02_cue_aligned_eeg.png",

        zero_label=
            "cue",

        show=
            args.show,
    )

    plot_condition_averages(
        epochs=
            movement_epochs,

        title=
            "Movement-Onset-Aligned EEG — LEFT vs RIGHT",

        output_file=
            output_dir
            /
            "03_movement_aligned_eeg.png",

        zero_label=
            "movement onset",

        show=
            args.show,
    )

    plot_movement_psd(
        movement_epochs=
            movement_epochs,

        sfreq=
            recording.sfreq,

        output_file=
            output_dir
            /
            "04_movement_psd.png",

        show=
            args.show,
    )

    plot_tfr_condition(
        epochs=
            cue_epochs,

        sfreq=
            recording.sfreq,

        side=
            "LEFT",

        alignment_name=
            "Cue",

        zero_label=
            "cue",

        output_file=
            output_dir
            /
            "05_cue_tfr_left.png",

        show=
            args.show,
    )

    plot_tfr_condition(
        epochs=
            cue_epochs,

        sfreq=
            recording.sfreq,

        side=
            "RIGHT",

        alignment_name=
            "Cue",

        zero_label=
            "cue",

        output_file=
            output_dir
            /
            "06_cue_tfr_right.png",

        show=
            args.show,
    )

    plot_tfr_condition(
        epochs=
            movement_epochs,

        sfreq=
            recording.sfreq,

        side=
            "LEFT",

        alignment_name=
            "Movement",

        zero_label=
            "movement onset",

        output_file=
            output_dir
            /
            "07_movement_tfr_left.png",

        show=
            args.show,
    )

    plot_tfr_condition(
        epochs=
            movement_epochs,

        sfreq=
            recording.sfreq,

        side=
            "RIGHT",

        alignment_name=
            "Movement",

        zero_label=
            "movement onset",

        output_file=
            output_dir
            /
            "08_movement_tfr_right.png",

        show=
            args.show,
    )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    write_summary(
        output_file=
            output_dir
            /
            "analysis_summary.txt",

        xdf_file=
            xdf_file,

        validation_csv=
            validation_csv,

        validation=
            validation,

        valid_trials=
            valid_trials,

        recording=
            recording,

        cue_timing_errors_ms=
            cue_result.timing_errors_ms,

        movement_timing_errors_ms=
            movement_result.timing_errors_ms,
    )

    print()
    print(
        "========================================"
    )
    print(
        " ANALYSIS COMPLETE"
    )
    print(
        "========================================"
    )
    print()

    print(
        f"Output folder: "
        f"{output_dir}"
    )

    print()

    print(
        "Generated:"
    )

    print(
        "  validated_trials_used.csv"
    )

    print(
        "  cue_aligned-epo.fif"
    )

    print(
        "  movement_aligned-epo.fif"
    )

    print(
        "  01_reaction_times.png"
    )

    print(
        "  02_cue_aligned_eeg.png"
    )

    print(
        "  03_movement_aligned_eeg.png"
    )

    print(
        "  04_movement_psd.png"
    )

    print(
        "  05_cue_tfr_left.png"
    )

    print(
        "  06_cue_tfr_right.png"
    )

    print(
        "  07_movement_tfr_left.png"
    )

    print(
        "  08_movement_tfr_right.png"
    )

    print(
        "  analysis_summary.txt"
    )

    print()

    if args.show:

        plt.show()


if __name__ == "__main__":
    main()
