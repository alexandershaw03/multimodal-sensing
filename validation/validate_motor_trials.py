import re
import csv
from pathlib import Path

import numpy as np
import pyxdf


# =========================================================
# ATS MOTOR EXPERIMENT VALIDATOR
#
# Matches:
#
#   ATS_EXPERIMENT
#
# against:
#
#   ATS_VISION_EVENTS
#
# For each trial:
#
#   CUE
#    ↓
#   first observed movement
#    ↓
#   correct / wrong / missed
#
# Also independently checks DOWN movement.
# =========================================================


# =========================================================
# FILE
# =========================================================


# Replace "xxx" below with actual file path
XDF_FILE = Path(
    r" xxx "
)


# How much extra time after a nominal phase to allow PoseNet to recognise the movement.
VISION_GRACE_TIME = 0.75


# =========================================================
# LOAD XDF
# =========================================================

print()
print("========================================")
print(" ATS MOTOR EXPERIMENT VALIDATOR")
print("========================================")
print()

print("Loading:")
print(XDF_FILE)
print()


streams, header = pyxdf.load_xdf(
    str(XDF_FILE)
)


stream_map = {}


for stream in streams:

    name = stream["info"]["name"][0]

    stream_map[name] = stream


required = [
    "ATS_EXPERIMENT",
    "ATS_VISION_EVENTS",
    "ATS_EEG_RAW",
    "ATS_BODY_POSE"
]


for name in required:

    if name not in stream_map:

        raise RuntimeError(
            "Missing stream: {}".format(
                name
            )
        )


experiment_stream = stream_map[
    "ATS_EXPERIMENT"
]


vision_stream = stream_map[
    "ATS_VISION_EVENTS"
]


eeg_stream = stream_map[
    "ATS_EEG_RAW"
]


# =========================================================
# EXTRACT STREAM DATA
# =========================================================

experiment_times = np.asarray(
    experiment_stream["time_stamps"],
    dtype=float
)


experiment_values = [
    x[0]
    for x in experiment_stream["time_series"]
]


vision_times = np.asarray(
    vision_stream["time_stamps"],
    dtype=float
)


vision_values = [
    x[0]
    for x in vision_stream["time_series"]
]


eeg_start = float(
    eeg_stream["time_stamps"][0]
)


# =========================================================
# PARSERS
# =========================================================

cue_pattern = re.compile(
    r"TRIAL_(\d+)_CUE_(LEFT|RIGHT)"
)


config_pattern = re.compile(
    r"TRIAL_(\d+)_CONFIG"
    r"\|SIDE=(LEFT|RIGHT)"
    r"\|UP=([0-9.]+)"
    r"\|HOLD=([0-9.]+)"
    r"\|DOWN=([0-9.]+)"
    r"\|REST=([0-9.]+)"
)


phase_pattern = re.compile(
    r"TRIAL_(\d+)_(HOLD_START|DOWN_START|REST_START|COMPLETE)"
)


# =========================================================
# BUILD TRIAL DICTIONARY
# =========================================================

trials = {}


for marker, timestamp in zip(
    experiment_values,
    experiment_times
):

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------

    match = config_pattern.fullmatch(
        marker
    )

    if match:

        trial_num = int(
            match.group(1)
        )

        side = match.group(2)


        trials.setdefault(
            trial_num,
            {}
        )


        trials[trial_num].update(
            {
                "trial": trial_num,

                "side": side,

                "up_time":
                    float(match.group(3)),

                "hold_time":
                    float(match.group(4)),

                "down_time":
                    float(match.group(5)),

                "rest_time":
                    float(match.group(6))
            }
        )

        continue


    # -----------------------------------------------------
    # CUE
    # -----------------------------------------------------

    match = cue_pattern.fullmatch(
        marker
    )

    if match:

        trial_num = int(
            match.group(1)
        )

        side = match.group(2)


        trials.setdefault(
            trial_num,
            {}
        )


        trials[trial_num][
            "trial"
        ] = trial_num


        trials[trial_num][
            "side"
        ] = side


        trials[trial_num][
            "cue_time"
        ] = float(timestamp)


        continue


    # -----------------------------------------------------
    # OTHER PHASES
    # -----------------------------------------------------

    match = phase_pattern.fullmatch(
        marker
    )

    if match:

        trial_num = int(
            match.group(1)
        )

        phase = match.group(2)


        trials.setdefault(
            trial_num,
            {}
        )


        key_map = {

            "HOLD_START":
                "hold_start",

            "DOWN_START":
                "down_start",

            "REST_START":
                "rest_start",

            "COMPLETE":
                "complete_time"
        }


        trials[trial_num][
            key_map[phase]
        ] = float(timestamp)


# =========================================================
# FILTER VISION EVENTS
# =========================================================

movement_events = []


for name, timestamp in zip(
    vision_values,
    vision_times
):

    if name == "LEFT_MOVEMENT_START":

        movement_events.append(
            {
                "side": "LEFT",
                "time": float(timestamp),
                "event": name
            }
        )


    elif name == "RIGHT_MOVEMENT_START":

        movement_events.append(
            {
                "side": "RIGHT",
                "time": float(timestamp),
                "event": name
            }
        )


movement_events.sort(
    key=lambda x: x["time"]
)


# =========================================================
# FIND EVENTS INSIDE A WINDOW
# =========================================================

def events_between(
    start_time,
    end_time
):

    return [

        event

        for event in movement_events

        if (
            event["time"] >= start_time
            and
            event["time"] <= end_time
        )

    ]


# =========================================================
# MATCH TRIALS
# =========================================================

results = []


for trial_num in sorted(
    trials.keys()
):

    trial = trials[
        trial_num
    ]


    # Ignore malformed/incomplete records.
    if "cue_time" not in trial:

        continue


    side = trial[
        "side"
    ]


    opposite_side = (

        "RIGHT"

        if side == "LEFT"

        else "LEFT"

    )


    cue_time = trial[
        "cue_time"
    ]


    hold_start = trial.get(
        "hold_start"
    )


    down_start = trial.get(
        "down_start"
    )


    rest_start = trial.get(
        "rest_start"
    )


    # =====================================================
    # UP MOVEMENT WINDOW
    # =====================================================

    # Prefer actual HOLD timestamp from experiment.
    #
    # Otherwise fall back to configured UP duration.

    if hold_start is not None:

        up_window_end = (
            hold_start
            +
            VISION_GRACE_TIME
        )

    else:

        up_window_end = (
            cue_time
            +
            trial.get(
                "up_time",
                1.5
            )
            +
            VISION_GRACE_TIME
        )


    up_events = events_between(
        cue_time,
        up_window_end
    )


    # -----------------------------------------------------
    # First ANY movement
    # -----------------------------------------------------

    first_up_event = (

        up_events[0]

        if up_events

        else None

    )


    # -----------------------------------------------------
    # First EXPECTED movement
    # -----------------------------------------------------

    expected_up_events = [

        event

        for event in up_events

        if event["side"] == side

    ]


    expected_up_event = (

        expected_up_events[0]

        if expected_up_events

        else None

    )


    # =====================================================
    # CLASSIFY UP
    # =====================================================

    if first_up_event is None:

        up_status = "MISSED"


    elif first_up_event[
        "side"
    ] == side:

        up_status = "VALID"


    else:

        up_status = "WRONG_ARM"


    # -----------------------------------------------------
    # Reaction time
    # -----------------------------------------------------

    if expected_up_event is not None:

        reaction_time = (

            expected_up_event[
                "time"
            ]
            -
            cue_time

        )

    else:

        reaction_time = np.nan


    # =====================================================
    # DOWN / RETURN MOVEMENT WINDOW
    # =====================================================

    if (
        down_start is not None
        and
        rest_start is not None
    ):

        down_window_end = (

            rest_start
            +
            VISION_GRACE_TIME

        )


        down_events = events_between(
            down_start,
            down_window_end
        )


        expected_down_events = [

            event

            for event in down_events

            if event["side"] == side

        ]


        expected_down_event = (

            expected_down_events[0]

            if expected_down_events

            else None

        )


        first_down_event = (

            down_events[0]

            if down_events

            else None

        )


        if first_down_event is None:

            down_status = "MISSED"


        elif first_down_event[
            "side"
        ] == side:

            down_status = "VALID"


        else:

            down_status = "WRONG_ARM"


        if expected_down_event is not None:

            down_reaction = (

                expected_down_event[
                    "time"
                ]
                -
                down_start

            )

        else:

            down_reaction = np.nan


    else:

        down_status = "NO_PHASE_DATA"

        down_reaction = np.nan


    # =====================================================
    # OVERALL VALIDITY
    # =====================================================

    overall_valid = (

        up_status == "VALID"

    )


    # =====================================================
    # SAVE RESULT
    # =====================================================

    results.append(
        {
            "trial":
                trial_num,

            "side":
                side,

            "up_status":
                up_status,

            "reaction_time":
                reaction_time,

            "down_status":
                down_status,

            "down_reaction":
                down_reaction,

            "overall_valid":
                overall_valid,

            "cue_time":
                cue_time,

            "movement_time":
                (
                    expected_up_event[
                        "time"
                    ]

                    if expected_up_event
                    is not None

                    else np.nan
                ),

            "movement_time_eeg":
                (
                    expected_up_event[
                        "time"
                    ]
                    -
                    eeg_start

                    if expected_up_event
                    is not None

                    else np.nan
                ),

            "hold_time":
                trial.get(
                    "hold_time",
                    np.nan
                ),

            "rest_time":
                trial.get(
                    "rest_time",
                    np.nan
                )
        }
    )


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("========================================")
print(" TRIAL RESULTS")
print("========================================")
print()


for result in results:

    reaction = result[
        "reaction_time"
    ]


    if np.isnan(
        reaction
    ):

        reaction_text = "---"

    else:

        reaction_text = (
            "{:.3f}s".format(
                reaction
            )
        )


    if result[
        "overall_valid"
    ]:

        symbol = "OK"

    else:

        symbol = "XX"


    print(

        "Trial {:02d} | "
        "{:5s} | "
        "UP {:9s} | "
        "RT {:>7s} | "
        "DOWN {:9s} | "
        "{}"

        .format(

            result[
                "trial"
            ],

            result[
                "side"
            ],

            result[
                "up_status"
            ],

            reaction_text,

            result[
                "down_status"
            ],

            symbol

        )

    )


# =========================================================
# SUMMARY
# =========================================================

total = len(
    results
)


valid = sum(

    1

    for result in results

    if result[
        "overall_valid"
    ]

)


wrong = sum(

    1

    for result in results

    if result[
        "up_status"
    ] == "WRONG_ARM"

)


missed = sum(

    1

    for result in results

    if result[
        "up_status"
    ] == "MISSED"

)


left_valid = sum(

    1

    for result in results

    if (
        result[
            "side"
        ] == "LEFT"

        and

        result[
            "overall_valid"
        ]
    )

)


right_valid = sum(

    1

    for result in results

    if (
        result[
            "side"
        ] == "RIGHT"

        and

        result[
            "overall_valid"
        ]
    )

)


reaction_times = [

    result[
        "reaction_time"
    ]

    for result in results

    if (
        result[
            "overall_valid"
        ]
        and
        not np.isnan(
            result[
                "reaction_time"
            ]
        )
    )

]


print()
print("========================================")
print(" SUMMARY")
print("========================================")
print()


print(
    "Total trials:       {}".format(
        total
    )
)


print(
    "Valid trials:       {}".format(
        valid
    )
)


print(
    "Wrong-arm trials:   {}".format(
        wrong
    )
)


print(
    "Missed trials:      {}".format(
        missed
    )
)


print()


print(
    "Valid LEFT:         {}".format(
        left_valid
    )
)


print(
    "Valid RIGHT:        {}".format(
        right_valid
    )
)


if reaction_times:

    print()


    print(
        "Mean reaction:     {:.3f} s".format(
            np.mean(
                reaction_times
            )
        )
    )


    print(
        "Fastest reaction:  {:.3f} s".format(
            np.min(
                reaction_times
            )
        )
    )


    print(
        "Slowest reaction:  {:.3f} s".format(
            np.max(
                reaction_times
            )
        )
    )


# =========================================================
# SAVE VALIDATION CSV
# =========================================================

OUTPUT_CSV = (
    XDF_FILE.with_name(
        XDF_FILE.stem
        +
        "_validated_trials.csv"
    )
)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "trial",
            "side",
            "up_status",
            "reaction_time",
            "down_status",
            "down_reaction",
            "overall_valid",
            "cue_time",
            "movement_time",
            "movement_time_eeg",
            "hold_time",
            "rest_time"
        ]
    )


    writer.writeheader()


    for result in results:

        writer.writerow(
            result
        )


print()
print(
    "Validation CSV saved:"
)

print(
    OUTPUT_CSV
)

print()
