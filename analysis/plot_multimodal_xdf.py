import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# FILE
# =========================================================

# Change "xxx"'s with actual file path
XDF_FILE = Path(
    r" xxx "
)


EEG_CHANNELS = [
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4"
]


POSE_CHANNELS = [

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
    "POSENET_FPS"
]


# =========================================================
# LOAD
# =========================================================

print("Loading XDF...")

streams, header = pyxdf.load_xdf(
    str(XDF_FILE)
)


stream_map = {}

for stream in streams:

    name = stream["info"]["name"][0]

    stream_map[name] = stream

    print(
        "{}: {} samples".format(
            name,
            len(stream["time_stamps"])
        )
    )


# =========================================================
# CHECK STREAMS
# =========================================================

required = [
    "ATS_EEG_RAW",
    "ATS_MARKERS",
    "ATS_BODY_POSE",
    "ATS_VISION_EVENTS"
]


for name in required:

    if name not in stream_map:

        raise RuntimeError(
            "Missing stream: {}".format(name)
        )


eeg_stream = stream_map[
    "ATS_EEG_RAW"
]

manual_stream = stream_map[
    "ATS_MARKERS"
]

pose_stream = stream_map[
    "ATS_BODY_POSE"
]

vision_stream = stream_map[
    "ATS_VISION_EVENTS"
]


# =========================================================
# COMMON EXPERIMENT CLOCK
# =========================================================

# Use start of EEG recording as t = 0

t0 = eeg_stream[
    "time_stamps"
][0]


def relative_times(stream):

    return (
        np.asarray(
            stream["time_stamps"],
            dtype=float
        )
        -
        t0
    )


# =========================================================
# EEG
# =========================================================

eeg = np.asarray(
    eeg_stream["time_series"],
    dtype=float
)

eeg_time = relative_times(
    eeg_stream
)


# Cortex values are already microvolts in XDF
# Remove each channel mean (for display only)

eeg_display = (
    eeg
    -
    np.nanmean(
        eeg,
        axis=0
    )
)


# =========================================================
# POSE
# =========================================================

pose = np.asarray(
    pose_stream["time_series"],
    dtype=float
)

pose_time = relative_times(
    pose_stream
)


pose_index = {

    name: index

    for index, name
    in enumerate(
        POSE_CHANNELS
    )
}


def pose_channel(name):

    return pose[
        :,
        pose_index[name]
    ]


# Selected measurements

left_wrist_y = pose_channel(
    "L_WRIST_REL_Y"
)

right_wrist_y = pose_channel(
    "R_WRIST_REL_Y"
)

left_speed = pose_channel(
    "L_WRIST_SPEED"
)

right_speed = pose_channel(
    "R_WRIST_SPEED"
)

left_elbow = pose_channel(
    "L_ELBOW_ANGLE"
)

right_elbow = pose_channel(
    "R_ELBOW_ANGLE"
)


# =========================================================
# MARKERS
# =========================================================

manual_time = relative_times(
    manual_stream
)

manual_values = manual_stream[
    "time_series"
]


vision_time = relative_times(
    vision_stream
)

vision_values = vision_stream[
    "time_series"
]


# =========================================================
# CREATE FIGURE
# =========================================================

fig = plt.figure(
    figsize=(17, 13)
)


# ---------------------------------------------------------
# EEG
# ---------------------------------------------------------

ax_eeg = fig.add_subplot(
    5,
    1,
    1
)


spacing = 600


for i in range(5):

    offset = (
        4 - i
    ) * spacing

    ax_eeg.plot(
        eeg_time,
        eeg_display[:, i] + offset,
        linewidth=0.6
    )


ax_eeg.set_yticks(
    [
        4 * spacing,
        3 * spacing,
        2 * spacing,
        1 * spacing,
        0
    ]
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


# ---------------------------------------------------------
# WRIST POSITION
# ---------------------------------------------------------

ax_position = fig.add_subplot(
    5,
    1,
    2,
    sharex=ax_eeg
)


ax_position.plot(
    pose_time,
    left_wrist_y,
    label="Left wrist"
)

ax_position.plot(
    pose_time,
    right_wrist_y,
    label="Right wrist"
)


ax_position.set_ylabel(
    "Relative wrist Y\n(shoulder widths)"
)

ax_position.legend(
    loc="upper right"
)


# ---------------------------------------------------------
# WRIST SPEED
# ---------------------------------------------------------

ax_speed = fig.add_subplot(
    5,
    1,
    3,
    sharex=ax_eeg
)


ax_speed.plot(
    pose_time,
    left_speed,
    label="Left"
)

ax_speed.plot(
    pose_time,
    right_speed,
    label="Right"
)


ax_speed.set_ylabel(
    "Wrist speed\n(body/s)"
)

ax_speed.legend(
    loc="upper right"
)


# ---------------------------------------------------------
# ELBOW ANGLES
# ---------------------------------------------------------

ax_angle = fig.add_subplot(
    5,
    1,
    4,
    sharex=ax_eeg
)


ax_angle.plot(
    pose_time,
    left_elbow,
    label="Left"
)

ax_angle.plot(
    pose_time,
    right_elbow,
    label="Right"
)


ax_angle.set_ylabel(
    "Elbow angle\n(deg)"
)

ax_angle.legend(
    loc="upper right"
)


# ---------------------------------------------------------
# EVENT TIMELINE
# ---------------------------------------------------------

ax_events = fig.add_subplot(
    5,
    1,
    5,
    sharex=ax_eeg
)


ax_events.set_ylim(
    -0.5,
    1.5
)

ax_events.set_yticks(
    [0, 1]
)

ax_events.set_yticklabels(
    [
        "Manual",
        "Vision"
    ]
)


# Manual markers

for marker, timestamp in zip(
    manual_values,
    manual_time
):

    name = marker[0]

    ax_events.scatter(
        timestamp,
        0
    )

    ax_events.text(
        timestamp,
        0.08,
        name,
        rotation=90,
        fontsize=8,
        verticalalignment="bottom"
    )


# Vision events

for marker, timestamp in zip(
    vision_values,
    vision_time
):

    name = marker[0]

    ax_events.scatter(
        timestamp,
        1
    )

    ax_events.text(
        timestamp,
        1.08,
        name,
        rotation=90,
        fontsize=8,
        verticalalignment="bottom"
    )


ax_events.set_xlabel(
    "Time from EEG recording start (seconds)"
)


# =========================================================
# DRAW EVENT LINES THROUGH ALL SIGNAL PLOTS
# =========================================================

signal_axes = [
    ax_eeg,
    ax_position,
    ax_speed,
    ax_angle
]


for timestamp in manual_time:

    for ax in signal_axes:

        ax.axvline(
            timestamp,
            linestyle="--",
            linewidth=0.8,
            alpha=0.4
        )


for timestamp in vision_time:

    for ax in signal_axes:

        ax.axvline(
            timestamp,
            linestyle=":",
            linewidth=0.9,
            alpha=0.5
        )


# =========================================================
# FORMATTING
# =========================================================

for ax in [
    ax_eeg,
    ax_position,
    ax_speed,
    ax_angle,
    ax_events
]:

    ax.grid(
        True,
        axis="x",
        alpha=0.2
    )


plt.tight_layout()

plt.show()
