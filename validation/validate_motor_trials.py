"""
Behaviour validator, for ATS motor-response experiment.
Compares LEFT / RIGHT motor trials from ATS_EXPERIMENT, against independently detected physical movement events from ATS_VISION_EVENTS.

For each trial:

    experiment cue
          │
          ▼
    vision-derived movement
          │
          ├── expected arm first  -> VALID
          ├── opposite arm first  -> WRONG_ARM
          └── no movement         -> MISSED

The DOWN / return movement is evaluated independently.

The resulting CSV is consumed by validated EEG analysis pipeline.

Expected XDF streams
--------------------
ATS_EXPERIMENT
    Timestamped experiment configuration and phase markers.

ATS_VISION_EVENTS
    Vision-derived movement events, such as LEFT_MOVEMENT_START and RIGHT_MOVEMENT_START.

ATS_EEG_RAW
    Used only to establish EEG recording start time, for legacy movement_time_eeg output field.

Usage
-----
python validation/validate_motor_trials.py recording.xdf

Optional:
python validation/validate_motor_trials.py recording.xdf \
    --output validated_trials.csv \
    --vision-grace 0.75
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import pyxdf


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_VISION_GRACE_TIME = 0.75

EXPERIMENT_STREAM_NAME = "ATS_EXPERIMENT"
VISION_STREAM_NAME = "ATS_VISION_EVENTS"
EEG_STREAM_NAME = "ATS_EEG_RAW"


# ============================================================================
# MARKER PATTERNS
# ============================================================================

CUE_PATTERN = re.compile(
    r"TRIAL_(\d+)_CUE_(LEFT|RIGHT)"
)

CONFIG_PATTERN = re.compile(
    r"TRIAL_(\d+)_CONFIG"
    r"\|SIDE=(LEFT|RIGHT)"
    r"\|UP=([0-9.]+)"
    r"\|HOLD=([0-9.]+)"
    r"\|DOWN=([0-9.]+)"
    r"\|REST=([0-9.]+)"
)

PHASE_PATTERN = re.compile(
    r"TRIAL_(\d+)_(HOLD_START|DOWN_START|REST_START|COMPLETE)"
)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class TrialRecord:
    """Experiment timing info, reconstructed from LSL markers."""

    trial: int
    side: str | None = None

    up_time: float | None = None
    hold_time: float | None = None
    down_time: float | None = None
    rest_time: float | None = None

    cue_time: float | None = None
    hold_start: float | None = None
    down_start: float | None = None
    rest_start: float | None = None
    complete_time: float | None = None


@dataclass(frozen=True)
class MovementEvent:
    """One independently detected arm-movement event."""

    side: str
    time: float
    event: str


@dataclass
class ValidationResult:
    """Behavioural validation result, for one commanded trial."""

    trial: int
    side: str

    up_status: str
    reaction_time: float

    down_status: str
    down_reaction: float

    overall_valid: bool

    cue_time: float
    movement_time: float
    movement_time_eeg: float

    hold_time: float
    rest_time: float

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-compatible representation."""

        return {
            "trial": self.trial,
            "side": self.side,
            "up_status": self.up_status,
            "reaction_time": self.reaction_time,
            "down_status": self.down_status,
            "down_reaction": self.down_reaction,
            "overall_valid": self.overall_valid,
            "cue_time": self.cue_time,
            "movement_time": self.movement_time,
            "movement_time_eeg": self.movement_time_eeg,
            "hold_time": self.hold_time,
            "rest_time": self.rest_time,
        }


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

    timestamps = stream.get(
        "time_stamps",
        [],
    )

    if len(timestamps) == 0:
        raise RuntimeError(
            f"XDF stream '{name}' contains no samples."
        )

    return stream


def first_channel_values(
    stream: dict,
) -> list[str]:
    """
    Extract the first channel of a string/event XDF stream.
    """

    values = []

    for sample in stream["time_series"]:
        value = sample[0]

        if isinstance(value, bytes):
            value = value.decode(
                "utf-8",
                errors="replace",
            )

        values.append(
            str(value)
        )

    return values


# ============================================================================
# EXPERIMENT MARKER PARSING
# ============================================================================


def get_trial(
    trials: dict[int, TrialRecord],
    trial_number: int,
) -> TrialRecord:
    """Get or create a TrialRecord."""

    if trial_number not in trials:
        trials[trial_number] = TrialRecord(
            trial=trial_number
        )

    return trials[trial_number]


def parse_experiment_trials(
    experiment_stream: dict,
) -> dict[int, TrialRecord]:
    """
    Reconstruct trial timing from ATS_EXPERIMENT markers.
    """

    timestamps = experiment_stream[
        "time_stamps"
    ]

    markers = first_channel_values(
        experiment_stream
    )

    trials: dict[int, TrialRecord] = {}

    phase_attribute = {
        "HOLD_START": "hold_start",
        "DOWN_START": "down_start",
        "REST_START": "rest_start",
        "COMPLETE": "complete_time",
    }

    for marker, timestamp_raw in zip(
        markers,
        timestamps,
    ):
        timestamp = float(timestamp_raw)

        # ------------------------------------------------------------------
        # Trial configuration
        # ------------------------------------------------------------------

        match = CONFIG_PATTERN.fullmatch(
            marker
        )

        if match:
            trial_number = int(
                match.group(1)
            )

            trial = get_trial(
                trials,
                trial_number,
            )

            trial.side = match.group(2)
            trial.up_time = float(match.group(3))
            trial.hold_time = float(match.group(4))
            trial.down_time = float(match.group(5))
            trial.rest_time = float(match.group(6))

            continue

        # ------------------------------------------------------------------
        # Trial cue
        # ------------------------------------------------------------------

        match = CUE_PATTERN.fullmatch(
            marker
        )

        if match:
            trial_number = int(
                match.group(1)
            )

            trial = get_trial(
                trials,
                trial_number,
            )

            trial.side = match.group(2)
            trial.cue_time = timestamp

            continue

        # ------------------------------------------------------------------
        # Phase transitions
        # ------------------------------------------------------------------

        match = PHASE_PATTERN.fullmatch(
            marker
        )

        if match:
            trial_number = int(
                match.group(1)
            )

            phase = match.group(2)

            trial = get_trial(
                trials,
                trial_number,
            )

            setattr(
                trial,
                phase_attribute[phase],
                timestamp,
            )

    return trials


# ============================================================================
# VISION EVENT PARSING
# ============================================================================


def parse_movement_events(
    vision_stream: dict,
) -> list[MovementEvent]:
    """Extract supported movement-start events from ATS_VISION_EVENTS."""

    timestamps = vision_stream[
        "time_stamps"
    ]

    event_names = first_channel_values(
        vision_stream
    )

    side_map = {
        "LEFT_MOVEMENT_START": "LEFT",
        "RIGHT_MOVEMENT_START": "RIGHT",
    }

    events = []

    for event_name, timestamp in zip(
        event_names,
        timestamps,
    ):
        side = side_map.get(
            event_name
        )

        if side is None:
            continue

        events.append(
            MovementEvent(
                side=side,
                time=float(timestamp),
                event=event_name,
            )
        )

    events.sort(
        key=lambda event: event.time
    )

    return events


def events_between(
    movement_events: list[MovementEvent],
    start_time: float,
    end_time: float,
) -> list[MovementEvent]:
    """Return movement events inside inclusive time window."""

    return [
        event
        for event in movement_events
        if start_time
        <= event.time
        <= end_time
    ]


# ============================================================================
# MOVEMENT CLASSIFICATION
# ============================================================================


def classify_window(
    events: list[MovementEvent],
    expected_side: str,
    reference_time: float,
) -> tuple[
    str,
    MovementEvent | None,
    float,
]:
    """
    Classify one movement window.

    Classification based on the FIRST observed arm movement:

    no movement
        -> MISSED

    expected arm first
        -> VALID

    opposite arm first
        -> WRONG_ARM

    Reaction time is measured to the first expected-side movement, if one exists in window. 
    This preserves the behaviour of the original validator.
    """

    if not events:
        return (
            "MISSED",
            None,
            math.nan,
        )

    first_event = events[0]

    expected_event = next(
        (
            event
            for event in events
            if event.side == expected_side
        ),
        None,
    )

    if first_event.side == expected_side:
        status = "VALID"
    else:
        status = "WRONG_ARM"

    reaction_time = (
        expected_event.time
        - reference_time
        if expected_event is not None
        else math.nan
    )

    return (
        status,
        expected_event,
        reaction_time,
    )


# ============================================================================
# TRIAL VALIDATION
# ============================================================================


def validate_trial(
    trial: TrialRecord,
    movement_events: list[MovementEvent],
    eeg_start: float,
    vision_grace_time: float,
    require_down_valid: bool,
) -> ValidationResult | None:
    """Validate one reconstructed experiment trial."""

    if (
        trial.cue_time is None
        or trial.side not in {"LEFT", "RIGHT"}
    ):
        print(
            f"Warning: skipping incomplete trial "
            f"{trial.trial}."
        )
        return None

    side = trial.side
    cue_time = trial.cue_time

    # ----------------------------------------------------------------------
    # UP movement window
    # ----------------------------------------------------------------------

    if trial.hold_start is not None:
        up_window_end = (
            trial.hold_start
            + vision_grace_time
        )

    else:
        fallback_up_duration = (
            trial.up_time
            if trial.up_time is not None
            else 1.5
        )

        up_window_end = (
            cue_time
            + fallback_up_duration
            + vision_grace_time
        )

    up_events = events_between(
        movement_events,
        cue_time,
        up_window_end,
    )

    (
        up_status,
        expected_up_event,
        reaction_time,
    ) = classify_window(
        events=up_events,
        expected_side=side,
        reference_time=cue_time,
    )

    # ----------------------------------------------------------------------
    # DOWN / return movement window
    # ----------------------------------------------------------------------

    if (
        trial.down_start is not None
        and trial.rest_start is not None
    ):
        down_window_end = (
            trial.rest_start
            + vision_grace_time
        )

        down_events = events_between(
            movement_events,
            trial.down_start,
            down_window_end,
        )

        (
            down_status,
            _expected_down_event,
            down_reaction,
        ) = classify_window(
            events=down_events,
            expected_side=side,
            reference_time=trial.down_start,
        )

    else:
        down_status = "NO_PHASE_DATA"
        down_reaction = math.nan

    # ----------------------------------------------------------------------
    # Overall behavioural validity
    # ----------------------------------------------------------------------

    if require_down_valid:
        overall_valid = (
            up_status == "VALID"
            and down_status == "VALID"
        )

    else:
        # Backward-compatible behaviour: the original analysis-defined trial validity from UP response.
        overall_valid = (
            up_status == "VALID"
        )

    # ----------------------------------------------------------------------
    # Timing outputs
    # ----------------------------------------------------------------------

    if expected_up_event is not None:
        movement_time = (
            expected_up_event.time
        )

        movement_time_eeg = (
            expected_up_event.time
            - eeg_start
        )

    else:
        movement_time = math.nan
        movement_time_eeg = math.nan

    return ValidationResult(
        trial=trial.trial,
        side=side,
        up_status=up_status,
        reaction_time=reaction_time,
        down_status=down_status,
        down_reaction=down_reaction,
        overall_valid=overall_valid,
        cue_time=cue_time,
        movement_time=movement_time,
        movement_time_eeg=movement_time_eeg,
        hold_time=(
            trial.hold_time
            if trial.hold_time is not None
            else math.nan
        ),
        rest_time=(
            trial.rest_time
            if trial.rest_time is not None
            else math.nan
        ),
    )


def validate_trials(
    trials: dict[int, TrialRecord],
    movement_events: list[MovementEvent],
    eeg_start: float,
    vision_grace_time: float,
    require_down_valid: bool,
) -> list[ValidationResult]:
    """Validate all complete experiment trials."""

    results = []

    for trial_number in sorted(trials):
        result = validate_trial(
            trial=trials[trial_number],
            movement_events=movement_events,
            eeg_start=eeg_start,
            vision_grace_time=vision_grace_time,
            require_down_valid=require_down_valid,
        )

        if result is not None:
            results.append(result)

    return results


# ============================================================================
# OUTPUT
# ============================================================================


CSV_FIELDS = [
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
    "rest_time",
]


def reaction_text(
    value: float,
) -> str:
    """Format reaction time for terminal output."""

    if math.isnan(value):
        return "---"

    return f"{value:.3f}s"


def print_trial_results(
    results: list[ValidationResult],
) -> None:
    """Print trial-by-trial validation table."""

    print()
    print("========================================")
    print(" TRIAL RESULTS")
    print("========================================")
    print()

    for result in results:
        symbol = (
            "OK"
            if result.overall_valid
            else "XX"
        )

        print(
            f"Trial {result.trial:02d} | "
            f"{result.side:5s} | "
            f"UP {result.up_status:9s} | "
            f"RT {reaction_text(result.reaction_time):>7s} | "
            f"DOWN {result.down_status:13s} | "
            f"{symbol}"
        )


def print_summary(
    results: list[ValidationResult],
) -> None:
    """Print aggregate behavioural validation statistics."""

    total = len(results)

    valid = sum(
        result.overall_valid
        for result in results
    )

    wrong = sum(
        result.up_status == "WRONG_ARM"
        for result in results
    )

    missed = sum(
        result.up_status == "MISSED"
        for result in results
    )

    left_valid = sum(
        result.side == "LEFT"
        and result.overall_valid
        for result in results
    )

    right_valid = sum(
        result.side == "RIGHT"
        and result.overall_valid
        for result in results
    )

    reaction_times = [
        result.reaction_time
        for result in results
        if (
            result.overall_valid
            and not math.isnan(
                result.reaction_time
            )
        )
    ]

    print()
    print("========================================")
    print(" SUMMARY")
    print("========================================")
    print()

    print(f"Total trials:       {total}")
    print(f"Valid trials:       {valid}")
    print(f"Wrong-arm trials:   {wrong}")
    print(f"Missed trials:      {missed}")

    print()

    print(f"Valid LEFT:         {left_valid}")
    print(f"Valid RIGHT:        {right_valid}")

    if reaction_times:
        print()

        print(
            f"Mean reaction:      "
            f"{mean(reaction_times):.3f} s"
        )

        print(
            f"Fastest reaction:   "
            f"{min(reaction_times):.3f} s"
        )

        print(
            f"Slowest reaction:   "
            f"{max(reaction_times):.3f} s"
        )


def save_results(
    results: list[ValidationResult],
    output_csv: Path,
) -> None:
    """Write validation results using existing CSV schema."""

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                result.to_dict()
            )


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse validator command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate ATS LEFT / RIGHT motor trials against vision-derived movement events."
        )
    )

    parser.add_argument(
        "xdf",
        type=Path,
        help="Multimodal XDF recording to validate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output validation CSV. "
            "Defaults to <xdf_name>_validated_trials.csv."
        ),
    )

    parser.add_argument(
        "--vision-grace",
        type=float,
        default=DEFAULT_VISION_GRACE_TIME,
        help=(
            "Additional seconds allowed after a nominal movement phase for vision detection "
            f"(default: {DEFAULT_VISION_GRACE_TIME})."
        ),
    )

    parser.add_argument(
        "--require-down-valid",
        action="store_true",
        help=(
            "Require both UP and DOWN movements to validate a trial. "
            "By default, only UP response determines overall validity, matching original analysis behaviour."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()

    xdf_file = args.xdf.expanduser().resolve()

    if not xdf_file.is_file():
        raise FileNotFoundError(
            f"XDF file not found: {xdf_file}"
        )

    if args.vision_grace < 0:
        raise ValueError(
            "--vision-grace must be zero or greater."
        )

    output_csv = (
        args.output.expanduser().resolve()
        if args.output is not None
        else xdf_file.with_name(
            xdf_file.stem
            + "_validated_trials.csv"
        )
    )

    print()
    print("========================================")
    print(" ATS MOTOR EXPERIMENT VALIDATOR")
    print("========================================")
    print()
    print(f"XDF:           {xdf_file}")
    print(
        f"Vision grace:  "
        f"{args.vision_grace:.3f} s"
    )

    validity_policy = (
        "UP + DOWN"
        if args.require_down_valid
        else "UP response"
    )

    print(
        f"Validity rule: {validity_policy}"
    )
    print()

    # ----------------------------------------------------------------------
    # Load recording
    # ----------------------------------------------------------------------

    streams, _header = pyxdf.load_xdf(
        str(xdf_file)
    )

    stream_map = build_stream_map(
        streams
    )

    experiment_stream = require_stream(
        stream_map,
        EXPERIMENT_STREAM_NAME,
    )

    vision_stream = require_stream(
        stream_map,
        VISION_STREAM_NAME,
    )

    eeg_stream = require_stream(
        stream_map,
        EEG_STREAM_NAME,
    )

    # ----------------------------------------------------------------------
    # Reconstruct experiment and behaviour
    # ----------------------------------------------------------------------

    trials = parse_experiment_trials(
        experiment_stream
    )

    movement_events = parse_movement_events(
        vision_stream
    )

    eeg_start = float(
        eeg_stream["time_stamps"][0]
    )

    print(
        f"Experiment trials reconstructed: "
        f"{len(trials)}"
    )

    print(
        f"Vision movement events found:     "
        f"{len(movement_events)}"
    )

    if not trials:
        raise RuntimeError(
            "No motor trials could be reconstructed from ATS_EXPERIMENT."
        )

    if not movement_events:
        print(
            "Warning: no supported movement-start events were found in ATS_VISION_EVENTS."
        )

    # ----------------------------------------------------------------------
    # Validate
    # ----------------------------------------------------------------------

    results = validate_trials(
        trials=trials,
        movement_events=movement_events,
        eeg_start=eeg_start,
        vision_grace_time=args.vision_grace,
        require_down_valid=args.require_down_valid,
    )

    if not results:
        raise RuntimeError(
            "No complete motor trials were available for validation."
        )

    print_trial_results(
        results
    )

    print_summary(
        results
    )

    save_results(
        results,
        output_csv,
    )

    print()
    print(
        f"Validation CSV saved:\n"
        f"{output_csv}"
    )
    print()


if __name__ == "__main__":
    main()
