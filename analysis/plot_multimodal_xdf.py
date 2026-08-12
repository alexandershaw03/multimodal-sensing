"""
Plot synchronised multimodal neural-motor XDF recording.

The visualisation combines:
    ATS_EEG_RAW      - five-channel EEG
    ATS_BODY_POSE    - vision-derived upper-body kinematics
    ATS_EXPERIMENT   - automatic experiment markers, when present
    ATS_MARKERS      - manual markers, used as a fallback
    ATS_VISION_EVENTS - automatically detected vision events

All streams are plotted on a common timeline referenced to the first recorded EEG sample.

The pose reader is schema-aware. 
It first attempts to recover channel labels from XDF/LSL metadata, falling back to the known ATS V1 (24 channel) or V2 (32 channel) layouts when metadata is unavailable.

Usage
-----
python analysis/plot_multimodal_xdf.py recording.xdf

Optional:
python analysis/plot_multimodal_xdf.py recording.xdf \
    --output multimodal_timeline.png \
    --start 10 \
    --end 60 \
    --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxdf


# ============================================================================
# STREAM CONFIGURATION
# ============================================================================

EEG_STREAM_NAME = "ATS_EEG_RAW"
POSE_STREAM_NAME = "ATS_BODY_POSE"

EXPERIMENT_STREAM_NAME = "ATS_EXPERIMENT"
MANUAL_STREAM_NAME = "ATS_MARKERS"

VISION_STREAM_NAME = "ATS_VISION_EVENTS"


EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)


POSE_CHANNELS_V1 = (
    "L_SHOULDER_X",
    "L_SHOULDER_Y",
    "L_ELBOW_X",
    "L_ELBOW_Y",
    "L_WRIST_X",
    "L_WRIST_Y",
    "R_SHOULDER_X",
    "R_SHOULDER_Y",
    "R_ELBOW_X",
    "R_ELBOW_Y",
    "R_WRIST_X",
    "R_WRIST_Y",
    "L_ELBOW_ANGLE",
    "R_ELBOW_ANGLE",
    "L_WRIST_REL_X",
    "L_WRIST_REL_Y",
    "R_WRIST_REL_X",
    "R_WRIST_REL_Y",
    "L_WRIST_SPEED",
    "R_WRIST_SPEED",
    "L_WINDOW_TRAVEL",
    "R_WINDOW_TRAVEL",
    "SHOULDER_WIDTH",
    "POSENET_FPS",
)


POSE_CHANNELS_V2 = (
    *POSE_CHANNELS_V1,
    "L_NEUTRAL_DISTANCE",
    "R_NEUTRAL_DISTANCE",
    "L_TRIAL_PEAK_DISTANCE",
    "R_TRIAL_PEAK_DISTANCE",
    "L_TRIAL_ACTIVE",
    "R_TRIAL_ACTIVE",
    "L_TRIAL_READY",
    "R_TRIAL_READY",
)


REQUIRED_POSE_CHANNELS = (
    "L_WRIST_REL_Y",
    "R_WRIST_REL_Y",
    "L_WRIST_SPEED",
    "R_WRIST_SPEED",
    "L_ELBOW_ANGLE",
    "R_ELBOW_ANGLE",
)


# preserve spacing used by original visualisation.
DEFAULT_EEG_SPACING = 600.0


# ============================================================================
# XDF HELPERS
# ============================================================================


def stream_name(stream: dict) -> str:
    """Return LSL stream name stored in XDF stream."""

    return str(stream["info"]["name"][0])


def build_stream_map(streams: list[dict]) -> dict[str, dict]:
    """Index XDF streams by LSL stream name."""

    return {
        stream_name(stream): stream
        for stream in streams
    }


def require_stream(
    stream_map: dict[str, dict],
    name: str,
) -> dict:
    """Return required XDF stream or raise useful error."""

    if name not in stream_map:
        available = ", ".join(sorted(stream_map))

        raise RuntimeError(
            f"Required XDF stream '{name}' was not found.\n"
            f"Available streams: {available or 'none'}"
        )

    stream = stream_map[name]

    if len(stream.get("time_stamps", [])) == 0:
        raise RuntimeError(
            f"XDF stream '{name}' contains no samples."
        )

    return stream


def select_experiment_stream(
    stream_map: dict[str, dict],
) -> tuple[dict, str, str]:
    """
    Select experiment/annotation event stream.

    ATS_EXPERIMENT preferred, because it's produced by the current automatic motor-task application. 
    ATS_MARKERS is retained as a backwards-compatible fallback, for earlier/manual recordings.

    Returns
    -------
    stream - Selected XDF stream.
    row_label - Human-readable label for the event timeline.
    stream_name_value - LSL name of the selected stream.
    """

    if EXPERIMENT_STREAM_NAME in stream_map:
        stream = require_stream(
            stream_map,
            EXPERIMENT_STREAM_NAME,
        )

        return (
            stream,
            "Experiment",
            EXPERIMENT_STREAM_NAME,
        )

    if MANUAL_STREAM_NAME in stream_map:
        stream = require_stream(
            stream_map,
            MANUAL_STREAM_NAME,
        )

        return (
            stream,
            "Manual",
            MANUAL_STREAM_NAME,
        )

    available = ", ".join(sorted(stream_map))

    raise RuntimeError(
        "No experiment/annotation marker stream was found.\n"
        f"Expected '{EXPERIMENT_STREAM_NAME}' or "
        f"'{MANUAL_STREAM_NAME}'.\n"
        f"Available streams: {available or 'none'}"
    )


def decode_marker(value) -> str:
    """Decode marker value into normal Python string."""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def first_channel_values(stream: dict) -> list[str]:
    """Extract and decode first channel of marker stream."""

    values = []

    for sample in stream["time_series"]:
        if len(sample) == 0:
            values.append("")
            continue

        values.append(
            decode_marker(sample[0])
        )

    return values


def _first_text(value) -> str | None:
    """Return first string-like value, from nested XDF metadata."""

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)

            if text:
                return text

        return None

    return str(value)


def pose_channel_labels_from_metadata(
    pose_stream: dict,
) -> list[str]:
    """
    Recover channel labels from LSL metadata stored in XDF (pyxdf represents XML metadata hierarchy as nested dictionaries and lists). 
    ATS_BODY_POSE writes labels under:
        info -> desc -> channels -> channel -> label
    """

    try:
        desc = pose_stream["info"]["desc"][0]
        channels_container = desc["channels"][0]
        channel_entries = channels_container["channel"]

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        return []

    labels = []

    for channel in channel_entries:
        if not isinstance(channel, dict):
            return []

        label = _first_text(
            channel.get("label")
        )

        if not label:
            return []

        labels.append(label)

    return labels


def resolve_pose_channel_labels(
    pose_stream: dict,
    channel_count: int,
) -> tuple[str, ...]:
    """
    Determine pose schema used by recording.

    Metadata is preferred because they make the reader, independent of column position. 
    Known ATS layouts are used only as a compatibility fallback, for recordings without usable channel metadata.
    """

    metadata_labels = pose_channel_labels_from_metadata(
        pose_stream
    )

    if metadata_labels:
        if len(metadata_labels) != channel_count:
            raise RuntimeError(
                "ATS_BODY_POSE metadata channel-label count "
                f"({len(metadata_labels)}) does not match recorded "
                f"data width ({channel_count})."
            )

        if len(set(metadata_labels)) != len(
            metadata_labels
        ):
            raise RuntimeError(
                "ATS_BODY_POSE contains duplicate channel labels in XDF metadata."
            )

        return tuple(metadata_labels)

    if channel_count == len(POSE_CHANNELS_V1):
        print(
            "Pose metadata labels unavailable; "
            "using known ATS_BODY_POSE V1 24-channel layout."
        )

        return POSE_CHANNELS_V1

    if channel_count == len(POSE_CHANNELS_V2):
        print(
            "Pose metadata labels unavailable; "
            "using known ATS_BODY_POSE V2 32-channel layout."
        )

        return POSE_CHANNELS_V2

    raise RuntimeError(
        "Could not determine ATS_BODY_POSE channel layout. "
        f"Recording contains {channel_count} channels, "
        f"while known fallback layouts contain "
        f"{len(POSE_CHANNELS_V1)} and "
        f"{len(POSE_CHANNELS_V2)} channels."
    )


# ============================================================================
# COMMON TIMEBASE
# ============================================================================


def relative_times(
    stream: dict,
    reference_time: float,
) -> np.ndarray:
    """Convert XDF timestamps to seconds, relative to reference."""

    return (
        np.asarray(
            stream["time_stamps"],
            dtype=float,
        )
        -
        reference_time
    )


def time_mask(
    timestamps: np.ndarray,
    start: float | None,
    end: float | None,
) -> np.ndarray:
    """Return mask for optional, relative-time display window."""

    mask = np.ones(
        len(timestamps),
        dtype=bool,
    )

    if start is not None:
        mask &= timestamps >= start

    if end is not None:
        mask &= timestamps <= end

    return mask


# ============================================================================
# SIGNAL EXTRACTION
# ============================================================================


def extract_eeg(
    eeg_stream: dict,
    reference_time: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Extract five-channel EEG and remove channel means (for display only)
    """

    eeg = np.asarray(
        eeg_stream["time_series"],
        dtype=float,
    )

    timestamps = relative_times(
        eeg_stream,
        reference_time,
    )

    if eeg.ndim != 2:
        raise RuntimeError(
            f"Unexpected EEG array shape: {eeg.shape}"
        )

    if eeg.shape[0] != len(timestamps):
        raise RuntimeError(
            "EEG sample count does not match timestamp count."
        )

    if eeg.shape[1] != len(EEG_CHANNELS):
        raise RuntimeError(
            f"Expected {len(EEG_CHANNELS)} EEG channels "
            f"({', '.join(EEG_CHANNELS)}), "
            f"found {eeg.shape[1]}."
        )

    # Display-only centring (recorded values not modified)
    eeg_display = (
        eeg
        -
        np.nanmean(
            eeg,
            axis=0,
        )
    )

    return (
        eeg_display,
        timestamps,
    )


def extract_pose(
    pose_stream: dict,
    reference_time: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
    tuple[str, ...],
]:
    """Extract body-pose stream and resolve channel schema."""

    pose = np.asarray(
        pose_stream["time_series"],
        dtype=float,
    )

    timestamps = relative_times(
        pose_stream,
        reference_time,
    )

    if pose.ndim != 2:
        raise RuntimeError(
            f"Unexpected pose array shape: {pose.shape}"
        )

    if pose.shape[0] != len(timestamps):
        raise RuntimeError(
            "Pose sample count does not match timestamp count."
        )

    channel_labels = resolve_pose_channel_labels(
        pose_stream,
        pose.shape[1],
    )

    channel_index = {
        channel: index
        for index, channel
        in enumerate(channel_labels)
    }

    missing = [
        channel
        for channel in REQUIRED_POSE_CHANNELS
        if channel not in channel_index
    ]

    if missing:
        raise RuntimeError(
            "ATS_BODY_POSE is missing channels required by this visualisation: "
            + ", ".join(missing)
        )

    return (
        pose,
        timestamps,
        channel_index,
        channel_labels,
    )


def pose_channel(
    pose: np.ndarray,
    channel_index: dict[str, int],
    name: str,
) -> np.ndarray:
    """Return one named-pose channel."""

    if name not in channel_index:
        raise KeyError(
            f"Pose channel '{name}' is not present."
        )

    return pose[
        :,
        channel_index[name]
    ]


# ============================================================================
# EVENT EXTRACTION
# ============================================================================


def extract_events(
    stream: dict,
    reference_time: float,
) -> tuple[
    np.ndarray,
    list[str],
]:
    """Extract timestamps and decoded marker strings."""

    timestamps = relative_times(
        stream,
        reference_time,
    )

    values = first_channel_values(
        stream
    )

    if len(values) != len(timestamps):
        raise RuntimeError(
            f"Event count mismatch in stream "
            f"'{stream_name(stream)}'."
        )

    return (
        timestamps,
        values,
    )


# ============================================================================
# EVENT PLOTTING
# ============================================================================


def draw_event_row(
    ax: plt.Axes,
    timestamps: np.ndarray,
    labels: list[str],
    y: float,
    *,
    show_labels: bool,
) -> None:
    """Draws one row of event markers"""

    ax.scatter(
        timestamps,
        np.full(
            len(timestamps),
            y,
        ),
        s=28,
        zorder=3,
    )

    if not show_labels:
        return

    for timestamp, label in zip(
        timestamps,
        labels,
    ):
        ax.text(
            timestamp,
            y + 0.08,
            label,
            rotation=90,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="center",
        )


def draw_event_lines(
    axes: list[plt.Axes],
    experiment_times: np.ndarray,
    vision_times: np.ndarray,
) -> None:
    """
    Draw event timestamps, through all continuous-signal plots:
    Experiment/manual events  = dashed lines.
    Vision events             = dotted lines.
    """

    for timestamp in experiment_times:
        for ax in axes:
            ax.axvline(
                timestamp,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

    for timestamp in vision_times:
        for ax in axes:
            ax.axvline(
                timestamp,
                linestyle=":",
                linewidth=0.9,
                alpha=0.5,
            )


# ============================================================================
# FIGURE
# ============================================================================


def create_multimodal_figure(
    *,
    eeg: np.ndarray,
    eeg_time: np.ndarray,
    pose: np.ndarray,
    pose_time: np.ndarray,
    pose_index: dict[str, int],
    experiment_time: np.ndarray,
    experiment_labels: list[str],
    experiment_row_label: str,
    vision_time: np.ndarray,
    vision_labels: list[str],
    eeg_spacing: float,
    show_event_labels: bool,
) -> plt.Figure:
    """Create complete multimodal neural-motor timeline."""

    left_wrist_y = pose_channel(
        pose,
        pose_index,
        "L_WRIST_REL_Y",
    )

    right_wrist_y = pose_channel(
        pose,
        pose_index,
        "R_WRIST_REL_Y",
    )

    left_speed = pose_channel(
        pose,
        pose_index,
        "L_WRIST_SPEED",
    )

    right_speed = pose_channel(
        pose,
        pose_index,
        "R_WRIST_SPEED",
    )

    left_elbow = pose_channel(
        pose,
        pose_index,
        "L_ELBOW_ANGLE",
    )

    right_elbow = pose_channel(
        pose,
        pose_index,
        "R_ELBOW_ANGLE",
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(17, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.7,
                1.0,
                1.0,
                1.0,
                0.85,
            ]
        },
    )

    (
        ax_eeg,
        ax_position,
        ax_speed,
        ax_angle,
        ax_events,
    ) = axes

    # ----------------------------------------------------------------------
    # EEG
    # ----------------------------------------------------------------------

    offsets = np.arange(
        len(EEG_CHANNELS) - 1,
        -1,
        -1,
        dtype=float,
    ) * eeg_spacing

    for channel_index, offset in enumerate(
        offsets
    ):
        ax_eeg.plot(
            eeg_time,
            eeg[:, channel_index] + offset,
            linewidth=0.6,
        )

    ax_eeg.set_yticks(
        offsets
    )

    ax_eeg.set_yticklabels(
        EEG_CHANNELS
    )

    ax_eeg.set_ylabel(
        "EEG"
    )

    ax_eeg.set_title(
        "ATS Multimodal Neural-Motor Recording"
    )

    # ----------------------------------------------------------------------
    # WRIST POSITION
    # ----------------------------------------------------------------------

    ax_position.plot(
        pose_time,
        left_wrist_y,
        label="Left wrist",
    )

    ax_position.plot(
        pose_time,
        right_wrist_y,
        label="Right wrist",
    )

    ax_position.set_ylabel(
        "Relative wrist Y\n(shoulder widths)"
    )

    ax_position.legend(
        loc="upper right"
    )

    # ----------------------------------------------------------------------
    # WRIST SPEED
    # ----------------------------------------------------------------------

    ax_speed.plot(
        pose_time,
        left_speed,
        label="Left",
    )

    ax_speed.plot(
        pose_time,
        right_speed,
        label="Right",
    )

    ax_speed.set_ylabel(
        "Wrist speed\n(body/s)"
    )

    ax_speed.legend(
        loc="upper right"
    )

    # ----------------------------------------------------------------------
    # ELBOW ANGLES
    # ----------------------------------------------------------------------

    ax_angle.plot(
        pose_time,
        left_elbow,
        label="Left",
    )

    ax_angle.plot(
        pose_time,
        right_elbow,
        label="Right",
    )

    ax_angle.set_ylabel(
        "Elbow angle\n(deg)"
    )

    ax_angle.legend(
        loc="upper right"
    )

    # ----------------------------------------------------------------------
    # EVENT TIMELINE
    # ----------------------------------------------------------------------

    ax_events.set_ylim(
        -0.5,
        1.5,
    )

    ax_events.set_yticks(
        [0, 1]
    )

    ax_events.set_yticklabels(
        [
            experiment_row_label,
            "Vision",
        ]
    )

    draw_event_row(
        ax_events,
        experiment_time,
        experiment_labels,
        0.0,
        show_labels=show_event_labels,
    )

    draw_event_row(
        ax_events,
        vision_time,
        vision_labels,
        1.0,
        show_labels=show_event_labels,
    )

    ax_events.set_xlabel(
        "Time from EEG recording start (s)"
    )

    # ----------------------------------------------------------------------
    # EVENT LINES THROUGH CONTINUOUS SIGNALS
    # ----------------------------------------------------------------------

    draw_event_lines(
        [
            ax_eeg,
            ax_position,
            ax_speed,
            ax_angle,
        ],
        experiment_time,
        vision_time,
    )

    # ----------------------------------------------------------------------
    # FORMATTING
    # ----------------------------------------------------------------------

    for ax in axes:
        ax.grid(
            True,
            axis="x",
            alpha=0.2,
        )

    fig.tight_layout()

    return fig


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse multimodal plotting arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot synchronised ATS EEG, pose and event streams from XDF recording."
        )
    )

    parser.add_argument(
        "xdf",
        type=Path,
        help="Multimodal XDF recording.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. "
            "Defaults to <xdf_stem>_multimodal_timeline.png."
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "Optional plot start time in seconds, relative to EEG-recording start."
        ),
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help=(
            "Optional plot end time in seconds, relative to EEG-recording start."
        ),
    )

    parser.add_argument(
        "--eeg-spacing",
        type=float,
        default=DEFAULT_EEG_SPACING,
        help=(
            "Vertical EEG channel spacing in microvolts "
            f"(default: {DEFAULT_EEG_SPACING:g})."
        ),
    )

    parser.add_argument(
        "--no-event-labels",
        action="store_true",
        help=(
            "Hide vertical event text labels, while retaining event points and timing lines."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display figure interactively after saving."
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

    if (
        args.start is not None
        and args.end is not None
        and args.end <= args.start
    ):
        raise ValueError(
            "--end must be greater than --start."
        )

    if args.eeg_spacing <= 0:
        raise ValueError(
            "--eeg-spacing must be greater than zero."
        )

    output_file = (
        args.output
        .expanduser()
        .resolve()

        if args.output is not None

        else xdf_file.with_name(
            xdf_file.stem
            +
            "_multimodal_timeline.png"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("========================================")
    print(" ATS MULTIMODAL XDF PLOT")
    print("========================================")
    print()

    print(f"Loading: {xdf_file}")
    print()

    # ----------------------------------------------------------------------
    # LOAD XDF
    # ----------------------------------------------------------------------

    streams, _header = pyxdf.load_xdf(
        str(xdf_file)
    )

    stream_map = build_stream_map(
        streams
    )

    print("Streams:")

    for name in sorted(stream_map):
        print(
            f"  {name:<22} "
            f"{len(stream_map[name]['time_stamps'])} samples"
        )

    print()

    eeg_stream = require_stream(
        stream_map,
        EEG_STREAM_NAME,
    )

    pose_stream = require_stream(
        stream_map,
        POSE_STREAM_NAME,
    )

    (
        experiment_stream,
        experiment_row_label,
        experiment_stream_name,
    ) = select_experiment_stream(
        stream_map
    )

    vision_stream = require_stream(
        stream_map,
        VISION_STREAM_NAME,
    )

    # ----------------------------------------------------------------------
    # COMMON CLOCK
    # ----------------------------------------------------------------------

    reference_time = float(
        eeg_stream["time_stamps"][0]
    )

    (
        eeg,
        eeg_time,
    ) = extract_eeg(
        eeg_stream,
        reference_time,
    )

    (
        pose,
        pose_time,
        pose_index,
        pose_labels,
    ) = extract_pose(
        pose_stream,
        reference_time,
    )

    (
        experiment_time,
        experiment_labels,
    ) = extract_events(
        experiment_stream,
        reference_time,
    )

    (
        vision_time,
        vision_labels,
    ) = extract_events(
        vision_stream,
        reference_time,
    )

    # ----------------------------------------------------------------------
    # OPTIONAL TIME WINDOW
    # ----------------------------------------------------------------------

    eeg_keep = time_mask(
        eeg_time,
        args.start,
        args.end,
    )

    pose_keep = time_mask(
        pose_time,
        args.start,
        args.end,
    )

    experiment_keep = time_mask(
        experiment_time,
        args.start,
        args.end,
    )

    vision_keep = time_mask(
        vision_time,
        args.start,
        args.end,
    )

    eeg = eeg[
        eeg_keep
    ]

    eeg_time = eeg_time[
        eeg_keep
    ]

    pose = pose[
        pose_keep
    ]

    pose_time = pose_time[
        pose_keep
    ]

    experiment_time = experiment_time[
        experiment_keep
    ]

    experiment_labels = [
        label
        for label, keep
        in zip(
            experiment_labels,
            experiment_keep,
        )
        if keep
    ]

    vision_time = vision_time[
        vision_keep
    ]

    vision_labels = [
        label
        for label, keep
        in zip(
            vision_labels,
            vision_keep,
        )
        if keep
    ]

    if len(eeg_time) == 0:
        raise RuntimeError(
            "No EEG samples remain inside selected time window."
        )

    if len(pose_time) == 0:
        raise RuntimeError(
            "No pose samples remain inside selected time window."
        )

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------

    print(
        f"Pose schema:             {len(pose_labels)} channels"
    )

    print(
        f"Event stream:            {experiment_stream_name}"
    )

    print(
        f"EEG samples plotted:     {len(eeg_time)}"
    )

    print(
        f"Pose samples plotted:    {len(pose_time)}"
    )

    print(
        f"{experiment_row_label} events plotted: "
        f"{len(experiment_time)}"
    )

    print(
        f"Vision events plotted:   {len(vision_time)}"
    )

    print()

    # ----------------------------------------------------------------------
    # FIGURE
    # ----------------------------------------------------------------------

    figure = create_multimodal_figure(
        eeg=eeg,
        eeg_time=eeg_time,
        pose=pose,
        pose_time=pose_time,
        pose_index=pose_index,
        experiment_time=experiment_time,
        experiment_labels=experiment_labels,
        experiment_row_label=experiment_row_label,
        vision_time=vision_time,
        vision_labels=vision_labels,
        eeg_spacing=args.eeg_spacing,
        show_event_labels=(
            not args.no_event_labels
        ),
    )

    figure.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    print(
        f"Saved: {output_file}"
    )

    if args.show:
        plt.show()

    else:
        plt.close(
            figure
        )


if __name__ == "__main__":
    main()
