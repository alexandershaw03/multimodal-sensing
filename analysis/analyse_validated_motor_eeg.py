import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import mne
import matplotlib.pyplot as plt

from scipy.signal import welch


# =========================================================
# ATS VALIDATED MOTOR EEG ANALYSIS
#
# INPUTS:
#
#   1. Multimodal XDF
#   2. validated_trials.csv
#
# OUTPUTS:
#
#   - validated trial table
#   - cue-aligned MNE epochs
#   - movement-aligned MNE epochs
#   - reaction-time plot
#   - cue-aligned LEFT vs RIGHT averages
#   - movement-aligned LEFT vs RIGHT averages
#   - movement-aligned PSD
#   - cue-aligned ERD/ERS time-frequency plots
#   - movement-aligned ERD/ERS time-frequency plots
#
# =========================================================


# =========================================================
# FILES
# =========================================================

# Replace "xxx"'s with actual file paths
XDF_FILE = Path(
    r" xxx "
    r" xxx "
    r" xxx "
)

# Replace "xxx"'s with actual file paths
VALIDATION_CSV = Path(
    r" xxx "
    r" xxx "
    r" xxx "
)


OUTPUT_DIR = (
    XDF_FILE.parent
    /
    (
        XDF_FILE.stem
        +
        "_analysis"
    )
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# EEG SETTINGS
# =========================================================

EEG_CHANNELS = [
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4"
]


# Bandpass used for event-related analysis
FILTER_LOW = 1.0
FILTER_HIGH = 40.0


# Mains power interference
NOTCH_FREQ = 50.0


# =========================================================
# EPOCH SETTINGS
# =========================================================

CUE_TMIN = -2.0
CUE_TMAX = 3.0


MOVEMENT_TMIN = -2.0
MOVEMENT_TMAX = 2.0


# Baseline used for waveform visualisation and time-frequency ERD/ERS calculation.
BASELINE_START = -2.0
BASELINE_END = -1.0


# =========================================================
# TIME-FREQUENCY SETTINGS
# =========================================================

TFR_FREQS = np.arange(
    4.0,
    36.0,
    1.0
)


# Approx 0.5 s wavelet duration across frequencies
TFR_N_CYCLES = (
    TFR_FREQS / 2.0
)


# =========================================================
# LOAD XDF
# =========================================================

print()
print("========================================")
print(" ATS VALIDATED MOTOR EEG ANALYSIS")
print("========================================")
print()


print("Loading XDF:")
print(XDF_FILE)
print()


streams, header = pyxdf.load_xdf(
    str(XDF_FILE)
)


stream_map = {}


for stream in streams:

    name = stream["info"]["name"][0]

    stream_map[name] = stream


if "ATS_EEG_RAW" not in stream_map:

    raise RuntimeError(
        "ATS_EEG_RAW not found in XDF."
    )


eeg_stream = stream_map[
    "ATS_EEG_RAW"
]


# =========================================================
# LOAD VALIDATION RESULTS
# =========================================================

print("Loading validation CSV:")
print(VALIDATION_CSV)
print()


validation = pd.read_csv(
    VALIDATION_CSV
)


# Convert overall_valid properly, regardless of whether pandas reads it as bool or text.

validation[
    "valid_bool"
] = (

    validation[
        "overall_valid"
    ]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes"
        ]
    )

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


print(
    "Total commanded trials:",
    len(validation)
)


print(
    "Valid behavioural trials:",
    len(valid_trials)
)


print(
    "Valid LEFT:",
    sum(
        valid_trials[
            "side"
        ] == "LEFT"
    )
)


print(
    "Valid RIGHT:",
    sum(
        valid_trials[
            "side"
        ] == "RIGHT"
    )
)


print()


if len(valid_trials) == 0:

    raise RuntimeError(
        "No valid trials available."
    )


# Save exact table that this analysis used.

VALID_USED_FILE = (
    OUTPUT_DIR
    /
    "validated_trials_used.csv"
)


valid_trials.to_csv(
    VALID_USED_FILE,
    index=False
)


# =========================================================
# EXTRACT EEG
# =========================================================

eeg = np.asarray(
    eeg_stream[
        "time_series"
    ],
    dtype=float
)


eeg_timestamps = np.asarray(
    eeg_stream[
        "time_stamps"
    ],
    dtype=float
)


if eeg.ndim != 2:

    raise RuntimeError(
        "Unexpected EEG data shape."
    )


if eeg.shape[1] != len(
    EEG_CHANNELS
):

    raise RuntimeError(
        "Expected {} EEG channels, found {}.".format(
            len(
                EEG_CHANNELS
            ),
            eeg.shape[1]
        )
    )


SFREQ = float(
    eeg_stream[
        "info"
    ][
        "nominal_srate"
    ][0]
)


print(
    "EEG samples:",
    eeg.shape[0]
)


print(
    "EEG channels:",
    eeg.shape[1]
)


print(
    "Nominal EEG rate:",
    SFREQ,
    "Hz"
)


print()


# =========================================================
# CREATE MNE RAW
# =========================================================

# Cortex values stored in stream are microvolts but MNE expects volts.

eeg_volts = (

    eeg
    /
    1_000_000.0

).T


info = mne.create_info(
    ch_names=EEG_CHANNELS,
    sfreq=SFREQ,
    ch_types="eeg"
)


raw = mne.io.RawArray(
    eeg_volts,
    info,
    verbose=False
)


# =========================================================
# ELECTRODE LOCATIONS
# =========================================================

montage = (
    mne.channels.make_standard_montage(
        "standard_1020"
    )
)


raw.set_montage(
    montage,
    on_missing="warn"
)


# =========================================================
# FILTER EEG
# =========================================================

print(
    "Filtering EEG:"
)


print(
    "  notch:",
    NOTCH_FREQ,
    "Hz"
)


print(
    "  bandpass:",
    FILTER_LOW,
    "-",
    FILTER_HIGH,
    "Hz"
)


print()


raw_filtered = raw.copy()


raw_filtered.notch_filter(
    freqs=[
        NOTCH_FREQ
    ],
    picks="eeg",
    verbose=False
)


raw_filtered.filter(
    l_freq=FILTER_LOW,
    h_freq=FILTER_HIGH,
    picks="eeg",
    verbose=False
)


# =========================================================
# PRECISE XDF EVENT -> EEG SAMPLE MATCHING
# =========================================================

def nearest_eeg_sample(
    event_time
):
    """
    Find the actual EEG sample whose corrected XDF timestamp is closest to the event timestamp.

    Better than simply multiplying elapsed time by 128 because the real stream is not perfectly periodic.
    """

    insertion = np.searchsorted(
        eeg_timestamps,
        event_time
    )


    candidates = []


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

        return None, None


    best_index = min(

        candidates,

        key=lambda i: abs(
            eeg_timestamps[i]
            -
            event_time
        )

    )


    timing_error = (

        eeg_timestamps[
            best_index
        ]
        -
        event_time

    )


    return (
        best_index,
        timing_error
    )


# =========================================================
# BUILD MNE EPOCHS
# =========================================================

EVENT_ID = {
    "LEFT": 1,
    "RIGHT": 2
}


def build_epochs(
    alignment_name,
    timestamp_column,
    tmin,
    tmax
):
    """
    Build an MNE Epochs object using one timestamp column from the validated trial table.
    """

    events = []

    metadata_rows = []

    timing_errors = []


    print()
    print(
        "Building {}-aligned epochs...".format(
            alignment_name
        )
    )


    for _, row in valid_trials.iterrows():

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


        # ---------------------------------------------
        # Ensure complete epoch exists in XDF
        # ---------------------------------------------

        if (
            event_time + tmin
            <
            eeg_timestamps[0]
        ):

            print(
                "Skipping trial {}: "
                "too close to start.".format(
                    row[
                        "trial"
                    ]
                )
            )

            continue


        if (
            event_time + tmax
            >
            eeg_timestamps[-1]
        ):

            print(
                "Skipping trial {}: "
                "too close to end.".format(
                    row[
                        "trial"
                    ]
                )
            )

            continue


        # ---------------------------------------------
        # Timestamp -> actual EEG sample
        # ---------------------------------------------

        sample_index, error = (
            nearest_eeg_sample(
                event_time
            )
        )


        if sample_index is None:

            continue


        timing_errors.append(
            error
        )


        events.append(
            [
                int(
                    sample_index
                ),

                0,

                EVENT_ID[
                    side
                ]
            ]
        )


        metadata_rows.append(
            {
                "trial":
                    int(
                        row[
                            "trial"
                        ]
                    ),

                "side":
                    side,

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
                        error
                        *
                        1000.0
                    )
            }
        )


    if len(events) == 0:

        raise RuntimeError(
            "No usable {} events.".format(
                alignment_name
            )
        )


    events = np.asarray(
        events,
        dtype=int
    )


    metadata = pd.DataFrame(
        metadata_rows
    )


    epochs = mne.Epochs(
        raw_filtered,

        events,

        event_id=EVENT_ID,

        tmin=tmin,

        tmax=tmax,

        baseline=None,

        preload=True,

        metadata=metadata,

        reject_by_annotation=True,

        verbose=False
    )


    # ---------------------------------------------
    # Timing check
    # ---------------------------------------------

    timing_errors_ms = (
        np.asarray(
            timing_errors
        )
        *
        1000.0
    )


    print(
        "{} epochs created.".format(
            len(epochs)
        )
    )


    print(
        "Mean event/sample mismatch: "
        "{:.3f} ms".format(
            np.mean(
                np.abs(
                    timing_errors_ms
                )
            )
        )
    )


    print(
        "Worst event/sample mismatch: "
        "{:.3f} ms".format(
            np.max(
                np.abs(
                    timing_errors_ms
                )
            )
        )
    )


    return epochs


# =========================================================
# CUE EPOCHS
# =========================================================

cue_epochs = build_epochs(
    alignment_name="CUE",
    timestamp_column="cue_time",
    tmin=CUE_TMIN,
    tmax=CUE_TMAX
)


# =========================================================
# MOVEMENT EPOCHS
# =========================================================

movement_epochs = build_epochs(
    alignment_name="MOVEMENT",
    timestamp_column="movement_time",
    tmin=MOVEMENT_TMIN,
    tmax=MOVEMENT_TMAX
)


# =========================================================
# SAVE MNE EPOCH FILES
# =========================================================

CUE_EPOCH_FILE = (
    OUTPUT_DIR
    /
    "cue_aligned-epo.fif"
)


MOVEMENT_EPOCH_FILE = (
    OUTPUT_DIR
    /
    "movement_aligned-epo.fif"
)


cue_epochs.save(
    CUE_EPOCH_FILE,
    overwrite=True
)


movement_epochs.save(
    MOVEMENT_EPOCH_FILE,
    overwrite=True
)


print()
print(
    "Saved cue epochs:"
)

print(
    CUE_EPOCH_FILE
)


print()
print(
    "Saved movement epochs:"
)

print(
    MOVEMENT_EPOCH_FILE
)


# =========================================================
# REACTION TIME PLOT
# =========================================================

fig, ax = plt.subplots(
    figsize=(
        11,
        5
    )
)


trial_numbers = valid_trials[
    "trial"
].to_numpy()


reaction_times = valid_trials[
    "reaction_time"
].to_numpy(
    dtype=float
)


ax.scatter(
    trial_numbers,
    reaction_times,
    s=80
)


for trial, side, reaction in zip(
    trial_numbers,
    valid_trials[
        "side"
    ],
    reaction_times
):

    ax.text(
        trial,
        reaction + 0.025,
        side,
        ha="center",
        fontsize=9
    )


ax.axhline(
    np.mean(
        reaction_times
    ),
    linestyle="--",
    alpha=0.7,
    label=(
        "Mean = {:.3f} s".format(
            np.mean(
                reaction_times
            )
        )
    )
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


fig.savefig(
    OUTPUT_DIR
    /
    "01_reaction_times.png",
    dpi=160
)


# =========================================================
# BASELINE-CORRECTED EEG WAVEFORMS
# =========================================================

def get_baseline_corrected_data(
    epochs
):

    corrected = epochs.copy()


    corrected.apply_baseline(
        (
            BASELINE_START,
            BASELINE_END
        )
    )


    # Convert V -> µV for display.
    return (

        corrected.get_data()
        *
        1_000_000.0

    )


# =========================================================
# LEFT vs RIGHT AVERAGE EEG
# =========================================================

def plot_condition_averages(
    epochs,
    title,
    filename,
    zero_label
):

    times = epochs.times


    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,
        figsize=(
            13,
            12
        ),
        sharex=True
    )


    for channel_index, (
        channel,
        ax
    ) in enumerate(
        zip(
            EEG_CHANNELS,
            axes
        )
    ):

        for side in [
            "LEFT",
            "RIGHT"
        ]:

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
                get_baseline_corrected_data(
                    side_epochs
                )
            )


            average = np.mean(
                data[
                    :,
                    channel_index,
                    :
                ],
                axis=0
            )


            ax.plot(
                times,
                average,
                label=(
                    "{} (n={})".format(
                        side,
                        len(
                            side_epochs
                        )
                    )
                )
            )


        ax.axvline(
            0.0,
            linestyle="--",
            alpha=0.7
        )


        ax.axhline(
            0.0,
            linewidth=0.7,
            alpha=0.4
        )


        ax.set_ylabel(
            "{}\nµV".format(
                channel
            )
        )


        ax.grid(
            alpha=0.20
        )


        ax.legend(
            loc="upper right"
        )


    axes[-1].set_xlabel(
        "Time relative to {} (s)".format(
            zero_label
        )
    )


    fig.suptitle(
        title,
        fontsize=16
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )


    fig.savefig(
        OUTPUT_DIR
        /
        filename,
        dpi=160
    )


# =========================================================
# CUE-ALIGNED AVERAGE
# =========================================================

plot_condition_averages(
    cue_epochs,

    title=(
        "Cue-Aligned EEG — LEFT vs RIGHT"
    ),

    filename=(
        "02_cue_aligned_average_eeg.png"
    ),

    zero_label="cue"
)


# =========================================================
# MOVEMENT-ALIGNED AVERAGE
# =========================================================

plot_condition_averages(
    movement_epochs,

    title=(
        "Movement-Onset-Aligned EEG — LEFT vs RIGHT"
    ),

    filename=(
        "03_movement_aligned_average_eeg.png"
    ),

    zero_label="movement onset"
)


# =========================================================
# PSD
# =========================================================

def calculate_psd(
    epochs
):

    data = epochs.get_data()


    nperseg = min(
        256,
        data.shape[-1]
    )


    freqs, power = welch(
        data,
        fs=SFREQ,
        nperseg=nperseg,
        axis=-1
    )


    # V²/Hz -> µV²/Hz
    power = (
        power
        *
        1e12
    )


    keep = (
        (freqs >= 2.0)
        &
        (freqs <= 40.0)
    )


    return (
        freqs[
            keep
        ],

        power[
            ...,
            keep
        ]
    )


def plot_psd():

    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,
        figsize=(
            12,
            12
        ),
        sharex=True
    )


    for side in [
        "LEFT",
        "RIGHT"
    ]:

        side_epochs = movement_epochs[
            side
        ]


        if len(
            side_epochs
        ) == 0:

            continue


        freqs, power = calculate_psd(
            side_epochs
        )


        mean_power = np.mean(
            power,
            axis=0
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
                    "{} (n={})".format(
                        side,
                        len(
                            side_epochs
                        )
                    )
                )
            )


    for channel, ax in zip(
        EEG_CHANNELS,
        axes
    ):

        ax.set_ylabel(
            "{}\ndB".format(
                channel
            )
        )


        ax.grid(
            alpha=0.20
        )


        ax.legend(
            loc="upper right"
        )


    axes[-1].set_xlabel(
        "Frequency (Hz)"
    )


    fig.suptitle(
        "Movement-Aligned EEG Power Spectral Density",
        fontsize=16
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )


    fig.savefig(
        OUTPUT_DIR
        /
        "04_movement_psd.png",
        dpi=160
    )


plot_psd()


# =========================================================
# TIME-FREQUENCY / ERD-ERS
# =========================================================

def calculate_erds(
    epochs
):
    """
    Calculate Morlet power for every epoch independently, baseline-normalise each trial, then average.

    Output is percentage power change relative to baseline:

        0%     = baseline power
        <0%    = desynchronisation / reduced power
        >0%    = synchronisation / increased power
    """

    data = epochs.get_data()


    power = (
        mne.time_frequency.tfr_array_morlet(
            data,

            sfreq=SFREQ,

            freqs=TFR_FREQS,

            n_cycles=TFR_N_CYCLES,

            output="power",

            zero_mean=True,

            n_jobs=1,

            verbose=False
        )
    )


    times = epochs.times


    baseline_mask = (

        (times >= BASELINE_START)

        &

        (times <= BASELINE_END)

    )


    baseline_power = np.mean(
        power[
            ...,
            baseline_mask
        ],
        axis=-1,
        keepdims=True
    )


    # Avoid division by zero.
    baseline_power = np.maximum(
        baseline_power,
        1e-30
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


    # Average across trials.
    average_erds = np.mean(
        erds,
        axis=0
    )


    return (
        times,
        average_erds
    )


def plot_tfr_condition(
    epochs,
    side,
    alignment_name,
    zero_label,
    filename
):

    side_epochs = epochs[
        side
    ]


    if len(
        side_epochs
    ) == 0:

        return


    print(
        "Calculating {} {} TFR "
        "(n={})...".format(
            alignment_name,
            side,
            len(
                side_epochs
            )
        )
    )


    times, erds = calculate_erds(
        side_epochs
    )


    # Use a shared scale across channels within this figure.
    robust_limit = np.nanpercentile(
        np.abs(
            erds
        ),
        97
    )


    robust_limit = max(
        robust_limit,
        1.0
    )


    fig, axes = plt.subplots(
        len(
            EEG_CHANNELS
        ),
        1,
        figsize=(
            13,
            15
        ),
        sharex=True
    )


    image = None


    for channel_index, (
        channel,
        ax
    ) in enumerate(
        zip(
            EEG_CHANNELS,
            axes
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
                TFR_FREQS[-1]
            ],

            cmap="RdBu_r",

            vmin=-robust_limit,

            vmax=robust_limit
        )


        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.2
        )


        ax.set_ylabel(
            "{}\nHz".format(
                channel
            )
        )


    axes[-1].set_xlabel(
        "Time relative to {} (s)".format(
            zero_label
        )
    )


    fig.suptitle(
        "{}-Aligned {} ERD/ERS — n={}".format(
            alignment_name,
            side,
            len(
                side_epochs
            )
        ),
        fontsize=16
    )


    colorbar = fig.colorbar(
        image,
        ax=axes,
        pad=0.015
    )


    colorbar.set_label(
        "Power change from baseline (%)"
    )


    fig.subplots_adjust(
        left=0.10,
        right=0.88,
        bottom=0.06,
        top=0.94,
        hspace=0.18
    )


    fig.savefig(
        OUTPUT_DIR
        /
        filename,
        dpi=160
    )


# =========================================================
# CUE TFR
# =========================================================

plot_tfr_condition(
    cue_epochs,
    side="LEFT",
    alignment_name="Cue",
    zero_label="cue",
    filename="05_cue_tfr_LEFT.png"
)


plot_tfr_condition(
    cue_epochs,
    side="RIGHT",
    alignment_name="Cue",
    zero_label="cue",
    filename="06_cue_tfr_RIGHT.png"
)


# =========================================================
# MOVEMENT TFR
# =========================================================

plot_tfr_condition(
    movement_epochs,
    side="LEFT",
    alignment_name="Movement",
    zero_label="movement onset",
    filename="07_movement_tfr_LEFT.png"
)


plot_tfr_condition(
    movement_epochs,
    side="RIGHT",
    alignment_name="Movement",
    zero_label="movement onset",
    filename="08_movement_tfr_RIGHT.png"
)


# =========================================================
# SUMMARY FILE
# =========================================================

SUMMARY_FILE = (
    OUTPUT_DIR
    /
    "analysis_summary.txt"
)


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ATS VALIDATED MOTOR EEG ANALYSIS\n"
    )

    file.write(
        "================================\n\n"
    )


    file.write(
        "XDF:\n{}\n\n".format(
            XDF_FILE
        )
    )


    file.write(
        "Total commanded trials: {}\n".format(
            len(
                validation
            )
        )
    )


    file.write(
        "Valid trials: {}\n".format(
            len(
                valid_trials
            )
        )
    )


    file.write(
        "Valid LEFT: {}\n".format(
            sum(
                valid_trials[
                    "side"
                ]
                ==
                "LEFT"
            )
        )
    )


    file.write(
        "Valid RIGHT: {}\n".format(
            sum(
                valid_trials[
                    "side"
                ]
                ==
                "RIGHT"
            )
        )
    )


    file.write(
        "\nReaction time mean: {:.4f} s\n".format(
            np.mean(
                reaction_times
            )
        )
    )


    file.write(
        "Reaction time SD: {:.4f} s\n".format(
            np.std(
                reaction_times
            )
        )
    )


    file.write(
        "Reaction time min: {:.4f} s\n".format(
            np.min(
                reaction_times
            )
        )
    )


    file.write(
        "Reaction time max: {:.4f} s\n".format(
            np.max(
                reaction_times
            )
        )
    )


    file.write(
        "\nEEG filtering: {:.1f}-{:.1f} Hz\n".format(
            FILTER_LOW,
            FILTER_HIGH
        )
    )


    file.write(
        "Notch: {:.1f} Hz\n".format(
            NOTCH_FREQ
        )
    )


    file.write(
        "\nIMPORTANT:\n"
        "This analysis is exploratory. "
        "Behavioural validation does not constitute "
        "EEG artefact rejection.\n"
    )


print()
print("========================================")
print(" ANALYSIS COMPLETE")
print("========================================")
print()


print(
    "Output folder:"
)

print(
    OUTPUT_DIR
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
    "  02_cue_aligned_average_eeg.png"
)

print(
    "  03_movement_aligned_average_eeg.png"
)

print(
    "  04_movement_psd.png"
)

print(
    "  05_cue_tfr_LEFT.png"
)

print(
    "  06_cue_tfr_RIGHT.png"
)

print(
    "  07_movement_tfr_LEFT.png"
)

print(
    "  08_movement_tfr_RIGHT.png"
)

print(
    "  analysis_summary.txt"
)


print()
print(
    "Opening figures..."
)


plt.show()
