"""
Randomised LEFT / RIGHT motor-response experiment.

Creates hands-free Tkinter experiment interface, publishing all experiment events as timestamped LSL markers.

Trial sequence
--------------
     ARM UP
       ↓
      HOLD
       ↓
    ARM DOWN
       ↓
      REST

Features
--------
- user-configurable experiment duration (or, trials per arm)
- balanced random LEFT / RIGHT trial ordering, with no more than two identical sides consecutively (where possible)
- randomised HOLD and REST duration/s
- initial countdown for camera-positioning/user-orientation
- initial EEG baseline
- timestamped LSL experiment markers
- local CSV backup log
- ESC emergency abort

LSL stream
----------
Name:
    ATS_EXPERIMENT

Type:
    Markers

Local logs are written to, in order of precedence:

1. --log-dir supplied on the command line
2. <ATS_DATA_ROOT>/experiment_logs if ATS_DATA_ROOT is configured
3. <repository>/data/experiment_logs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from dotenv import load_dotenv
from pylsl import StreamInfo, StreamOutlet, local_clock


# ============================================================================
# PATHS / ENVIRONMENT
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")


# ============================================================================
# EXPERIMENT DEFAULTS
# ============================================================================

DEFAULT_TRIALS_PER_ARM = 10
DEFAULT_TARGET_MINUTES = 8.0

DEFAULT_UP_TIME = 1.5
DEFAULT_DOWN_TIME = 1.5

DEFAULT_HOLD_MIN = 2.0
DEFAULT_HOLD_MAX = 6.0

DEFAULT_REST_MIN = 4.0
DEFAULT_REST_MAX = 7.0

PREPARE_TIME = 10.0
INITIAL_BASELINE_TIME = 5.0

UPDATE_INTERVAL_MS = 50


# ============================================================================
# LSL
# ============================================================================

LSL_STREAM_NAME = "ATS_EXPERIMENT"
LSL_STREAM_TYPE = "Markers"
LSL_SOURCE_ID = "ats_motor_experiment_v2"


def create_experiment_outlet() -> StreamOutlet:
    """Create experiment-marker LSL outlet."""

    info = StreamInfo(
        name=LSL_STREAM_NAME,
        type=LSL_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=LSL_SOURCE_ID,
    )

    info.desc().append_child_value("manufacturer", "ATS")
    info.desc().append_child_value(
        "protocol",
        "Randomised Left-Right Motor Experiment",
    )
    info.desc().append_child_value("version", "2.0")

    return StreamOutlet(info)


# ============================================================================
# COLOURS
# ============================================================================

BG_DEFAULT = "#16181c"

BG_LEFT = "#17365d"
BG_RIGHT = "#6b2f1d"

BG_HOLD = "#665500"
BG_REST = "#243328"

BG_READY = "#333333"
BG_COMPLETE = "#174d2c"

TEXT_MAIN = "#ffffff"
TEXT_DIM = "#666a70"

TEXT_LEFT = "#8ec5ff"
TEXT_RIGHT = "#ffad8f"

CIRCLE_TRACK = "#595f68"
CIRCLE_ACTIVE = "#ffffff"


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass(frozen=True)
class Trial:
    """Timing and side configuration (for one motor trial)"""

    number: int
    side: str

    up_time: float
    hold_time: float
    down_time: float
    rest_time: float


# ============================================================================
# HELPERS
# ============================================================================


def random_balanced_sequence(trials_per_arm: int) -> list[str]:
    """
    Create balanced LEFT / RIGHT trial sequence.

    Prevents more than two identical sides appearing consecutively (when remaining trial counts make possible)
    """

    remaining = {
        "LEFT": trials_per_arm,
        "RIGHT": trials_per_arm,
    }

    sequence: list[str] = []

    while remaining["LEFT"] > 0 or remaining["RIGHT"] > 0:
        choices = [
            side
            for side in ("LEFT", "RIGHT")
            if remaining[side] > 0
        ]

        if (
            len(sequence) >= 2
            and sequence[-1] == sequence[-2]
        ):
            repeated_side = sequence[-1]

            alternatives = [
                side
                for side in choices
                if side != repeated_side
            ]

            if alternatives:
                choices = alternatives

        side = random.choice(choices)

        sequence.append(side)
        remaining[side] -= 1

    return sequence


def estimate_trials_per_arm(
    target_minutes: float,
    up_time: float,
    down_time: float,
    hold_min: float,
    hold_max: float,
    rest_min: float,
    rest_max: float,
) -> int:
    """
    Estimate balanced number of trials for experiment duration.
    """

    mean_hold = (hold_min + hold_max) / 2.0
    mean_rest = (rest_min + rest_max) / 2.0

    mean_trial_duration = (
        up_time
        + mean_hold
        + down_time
        + mean_rest
    )

    usable_seconds = max(
        1.0,
        target_minutes * 60.0
        - PREPARE_TIME
        - INITIAL_BASELINE_TIME,
    )

    estimated_total_trials = max(
        2,
        round(usable_seconds / mean_trial_duration),
    )

    # Total must remain even to preserve equal LEFT / RIGHT count
    if estimated_total_trials % 2:
        estimated_total_trials -= 1

    estimated_total_trials = max(
        2,
        estimated_total_trials,
    )

    return estimated_total_trials // 2


def resolve_log_directory(
    command_line_path: str | None,
) -> Path:
    """
    Resolve experiment log directory.

    Priority:
        1. --log-dir
        2. ATS_DATA_ROOT environment variable
        3. repository-local data/experiment_logs
    """

    if command_line_path:
        return Path(command_line_path).expanduser().resolve()

    data_root = os.getenv("ATS_DATA_ROOT")

    if data_root:
        return (
            Path(data_root)
            .expanduser()
            .resolve()
            / "experiment_logs"
        )

    return REPO_ROOT / "data" / "experiment_logs"


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class MotorExperimentApp:
    """Tkinter controller for complete motor-response experiment."""

    def __init__(
        self,
        root: tk.Tk,
        outlet: StreamOutlet,
        log_directory: Path,
    ) -> None:
        self.root = root
        self.outlet = outlet
        self.log_directory = log_directory

        self.root.title("ATS Motor Experiment")
        self.root.geometry("1000x760")
        self.root.configure(bg=BG_DEFAULT)

        # Experiment state
        self.running = False

        self.trials: list[Trial] = []
        self.current_trial_index = -1
        self.current_trial: Trial | None = None

        # Phase timing uses same LSL clock as marker timestamps.
        self.phase_name: str | None = None
        self.phase_start_clock: float | None = None
        self.phase_end_clock: float | None = None
        self.phase_duration: float | None = None

        # User-selected timing settings
        self.up_time = DEFAULT_UP_TIME
        self.down_time = DEFAULT_DOWN_TIME

        self.hold_min = DEFAULT_HOLD_MIN
        self.hold_max = DEFAULT_HOLD_MAX

        self.rest_min = DEFAULT_REST_MIN
        self.rest_max = DEFAULT_REST_MAX

        # CSV backup
        self.csv_file = None
        self.csv_writer = None
        self.log_path: Path | None = None

        # Emergency stop
        self.root.bind(
            "<Escape>",
            self.abort_experiment,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.request_close,
        )

        self.build_setup_screen()

    # ======================================================================
    # WINDOW UTILITIES
    # ======================================================================

    def clear_window(self) -> None:
        """Remove all widgets from the current screen."""

        for widget in self.root.winfo_children():
            widget.destroy()

    # ======================================================================
    # SETUP SCREEN
    # ======================================================================

    def build_setup_screen(self) -> None:
        """Build experiment configuration screen."""

        self.clear_window()

        self.running = False

        self.root.configure(bg=BG_DEFAULT)

        try:
            self.root.state("normal")
        except tk.TclError:
            pass

        tk.Label(
            self.root,
            text="ATS MOTOR EXPERIMENT",
            font=("Arial", 30, "bold"),
            fg=TEXT_MAIN,
            bg=BG_DEFAULT,
        ).pack(
            pady=(30, 10)
        )

        tk.Label(
            self.root,
            text="Hands-free randomised LEFT / RIGHT motor test",
            font=("Arial", 14),
            fg="#cccccc",
            bg=BG_DEFAULT,
        ).pack(
            pady=(0, 25)
        )

        frame = tk.Frame(
            self.root,
            bg=BG_DEFAULT,
        )

        frame.pack(
            padx=40,
            pady=10,
        )

        # ------------------------------------------------------------------
        # Experiment length
        # ------------------------------------------------------------------

        self.mode_var = tk.StringVar(
            value="trials"
        )

        tk.Label(
            frame,
            text="Experiment length:",
            font=("Arial", 13, "bold"),
            fg=TEXT_MAIN,
            bg=BG_DEFAULT,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=8,
        )

        tk.Radiobutton(
            frame,
            text="Trials per arm",
            variable=self.mode_var,
            value="trials",
            fg=TEXT_MAIN,
            bg=BG_DEFAULT,
            selectcolor=BG_DEFAULT,
            activebackground=BG_DEFAULT,
            activeforeground=TEXT_MAIN,
        ).grid(
            row=1,
            column=0,
            sticky="w",
        )

        self.trials_entry = tk.Entry(
            frame,
            width=10,
        )

        self.trials_entry.insert(
            0,
            str(DEFAULT_TRIALS_PER_ARM),
        )

        self.trials_entry.grid(
            row=1,
            column=1,
            padx=10,
        )

        tk.Radiobutton(
            frame,
            text="Approximate duration (minutes)",
            variable=self.mode_var,
            value="duration",
            fg=TEXT_MAIN,
            bg=BG_DEFAULT,
            selectcolor=BG_DEFAULT,
            activebackground=BG_DEFAULT,
            activeforeground=TEXT_MAIN,
        ).grid(
            row=2,
            column=0,
            sticky="w",
        )

        self.duration_entry = tk.Entry(
            frame,
            width=10,
        )

        self.duration_entry.insert(
            0,
            str(DEFAULT_TARGET_MINUTES),
        )

        self.duration_entry.grid(
            row=2,
            column=1,
            padx=10,
        )

        # ------------------------------------------------------------------
        # Movement timing
        # ------------------------------------------------------------------

        tk.Label(
            frame,
            text="Movement timing:",
            font=("Arial", 13, "bold"),
            fg=TEXT_MAIN,
            bg=BG_DEFAULT,
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(25, 8),
        )

        timing_fields = (
            ("UP cue duration (s)", DEFAULT_UP_TIME),
            ("DOWN cue duration (s)", DEFAULT_DOWN_TIME),
            ("Minimum HOLD time (s)", DEFAULT_HOLD_MIN),
            ("Maximum HOLD time (s)", DEFAULT_HOLD_MAX),
            ("Minimum REST time (s)", DEFAULT_REST_MIN),
            ("Maximum REST time (s)", DEFAULT_REST_MAX),
        )

        self.timing_entries: list[tk.Entry] = []

        for row, (label_text, default) in enumerate(
            timing_fields,
            start=4,
        ):
            tk.Label(
                frame,
                text=label_text,
                font=("Arial", 11),
                fg="#dddddd",
                bg=BG_DEFAULT,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                pady=3,
            )

            entry = tk.Entry(
                frame,
                width=10,
            )

            entry.insert(
                0,
                str(default),
            )

            entry.grid(
                row=row,
                column=1,
                padx=10,
                pady=3,
            )

            self.timing_entries.append(entry)

        tk.Button(
            self.root,
            text="START EXPERIMENT",
            font=("Arial", 16, "bold"),
            padx=30,
            pady=12,
            command=self.prepare_experiment,
        ).pack(
            pady=30
        )

        tk.Label(
            self.root,
            text=(
                "The LEFT / RIGHT order, HOLD duration and REST duration are randomised.\n"
                "A 10-second camera-positioning countdown occurs before baseline begins.\n"
                "No keyboard input required, once test starts.\n"
                "Press ESC to abort."
            ),
            font=("Arial", 11),
            fg="#aaaaaa",
            bg=BG_DEFAULT,
            justify="center",
        ).pack()

    # ======================================================================
    # PREPARE EXPERIMENT
    # ======================================================================

    def prepare_experiment(self) -> None:
        """Validate settings; create trials; initialise logging; start."""

        try:
            self._read_settings()

            if self.mode_var.get() == "trials":
                trials_per_arm = int(
                    self.trials_entry.get()
                )

                if trials_per_arm < 1:
                    raise ValueError

            else:
                target_minutes = float(
                    self.duration_entry.get()
                )

                if target_minutes <= 0:
                    raise ValueError

                trials_per_arm = estimate_trials_per_arm(
                    target_minutes=target_minutes,
                    up_time=self.up_time,
                    down_time=self.down_time,
                    hold_min=self.hold_min,
                    hold_max=self.hold_max,
                    rest_min=self.rest_min,
                    rest_max=self.rest_max,
                )

        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Please check experiment settings.",
            )
            return

        self.trials = self._create_trials(
            trials_per_arm
        )

        try:
            self.create_log_file()

        except OSError as exc:
            messagebox.showerror(
                "Log file error",
                f"Could not create experiment log:\n\n{exc}",
            )
            return

        self.build_experiment_screen()

        self.current_trial_index = -1
        self.current_trial = None

        self.running = True

        self.emit_marker(
            "EXPERIMENT_START"
        )

        self.start_phase(
            phase="PREPARE",
            duration=PREPARE_TIME,
            background=BG_READY,
            marker="PREPARE_START",
        )

    def _read_settings(self) -> None:
        """Read and validate movement timing fields."""

        (
            self.up_time,
            self.down_time,
            self.hold_min,
            self.hold_max,
            self.rest_min,
            self.rest_max,
        ) = [
            float(entry.get())
            for entry in self.timing_entries
        ]

        values = (
            self.up_time,
            self.down_time,
            self.hold_min,
            self.hold_max,
            self.rest_min,
            self.rest_max,
        )

        if any(value <= 0 for value in values):
            raise ValueError

        if self.hold_max < self.hold_min:
            raise ValueError

        if self.rest_max < self.rest_min:
            raise ValueError

    def _create_trials(
        self,
        trials_per_arm: int,
    ) -> list[Trial]:
        """Create randomised trial"""

        sequence = random_balanced_sequence(
            trials_per_arm
        )

        trials = []

        for number, side in enumerate(
            sequence,
            start=1,
        ):
            trials.append(
                Trial(
                    number=number,
                    side=side,
                    up_time=self.up_time,
                    hold_time=random.uniform(
                        self.hold_min,
                        self.hold_max,
                    ),
                    down_time=self.down_time,
                    rest_time=random.uniform(
                        self.rest_min,
                        self.rest_max,
                    ),
                )
            )

        return trials

    # ======================================================================
    # CSV BACKUP LOG
    # ======================================================================

    def create_log_file(self) -> None:
        """Create local timestamped experiment-event CSV."""

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.log_path = (
            self.log_directory
            / f"ATS_motor_experiment_v2_{timestamp}.csv"
        )

        self.csv_file = self.log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow(
            [
                "lsl_time",
                "marker",
                "trial",
                "side",
                "planned_up_s",
                "planned_hold_s",
                "planned_down_s",
                "planned_rest_s",
            ]
        )

        self.csv_file.flush()

        print()
        print(f"Experiment log: {self.log_path}")
        print()

    def close_log(self) -> None:
        """Flush and close current CSV log."""

        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()

        self.csv_file = None
        self.csv_writer = None

    # ======================================================================
    # EXPERIMENT DISPLAY
    # ======================================================================

    def build_experiment_screen(self) -> None:
        """Create full-screen experiment interface."""

        self.clear_window()

        self.root.configure(
            bg=BG_READY
        )

        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self.trial_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 20, "bold"),
            fg=TEXT_MAIN,
            bg=BG_READY,
        )

        self.trial_label.pack(
            pady=(20, 5)
        )

        # ------------------------------------------------------------------
        # Side indicators
        # ------------------------------------------------------------------

        self.side_frame = tk.Frame(
            self.root,
            bg=BG_READY,
        )

        self.side_frame.pack(
            fill="x",
            padx=50,
            pady=(5, 0),
        )

        self.left_indicator = tk.Label(
            self.side_frame,
            text="◀  LEFT",
            font=("Arial", 28, "bold"),
            fg=TEXT_DIM,
            bg=BG_READY,
        )

        self.left_indicator.pack(
            side="left",
            anchor="w",
        )

        self.right_indicator = tk.Label(
            self.side_frame,
            text="RIGHT  ▶",
            font=("Arial", 28, "bold"),
            fg=TEXT_DIM,
            bg=BG_READY,
        )

        self.right_indicator.pack(
            side="right",
            anchor="e",
        )

        # ------------------------------------------------------------------
        # Main cue
        # ------------------------------------------------------------------

        self.cue_label = tk.Label(
            self.root,
            text="GET READY",
            font=("Arial", 64, "bold"),
            fg=TEXT_MAIN,
            bg=BG_READY,
            justify="center",
        )

        self.cue_label.pack(
            expand=True
        )

        # ------------------------------------------------------------------
        # Countdown area
        # ------------------------------------------------------------------

        self.timer_frame = tk.Frame(
            self.root,
            bg=BG_READY,
        )

        self.timer_frame.pack(
            pady=5
        )

        self.countdown_canvas = tk.Canvas(
            self.timer_frame,
            width=190,
            height=190,
            bg=BG_READY,
            highlightthickness=0,
        )

        self.countdown_canvas.grid(
            row=0,
            column=0,
        )

        self.countdown_label = tk.Label(
            self.timer_frame,
            text="",
            font=("Arial", 28, "bold"),
            fg=TEXT_MAIN,
            bg=BG_READY,
        )

        self.countdown_label.grid(
            row=0,
            column=0,
        )

        self.countdown_canvas.grid_remove()

        # ------------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------------

        self.bottom_frame = tk.Frame(
            self.root,
            bg=BG_READY,
        )

        self.bottom_frame.pack(
            fill="x",
            padx=80,
            pady=(10, 25),
        )

        self.progress = ttk.Progressbar(
            self.bottom_frame,
            orient="horizontal",
            mode="determinate",
            maximum=len(self.trials),
        )

        self.progress.pack(
            fill="x"
        )

        self.progress_label = tk.Label(
            self.bottom_frame,
            text=f"0 / {len(self.trials)}",
            font=("Arial", 14),
            fg=TEXT_MAIN,
            bg=BG_READY,
        )

        self.progress_label.pack(
            pady=8
        )

    def set_active_side(
        self,
        side: str | None = None,
    ) -> None:
        """Highlight currently active arm."""

        if side == "LEFT":
            left_colour = TEXT_LEFT
            right_colour = TEXT_DIM

        elif side == "RIGHT":
            left_colour = TEXT_DIM
            right_colour = TEXT_RIGHT

        else:
            left_colour = TEXT_DIM
            right_colour = TEXT_DIM

        self.left_indicator.configure(
            fg=left_colour
        )

        self.right_indicator.configure(
            fg=right_colour
        )

    def set_background(
        self,
        colour: str,
    ) -> None:
        """Apply phase colour to interface."""

        self.root.configure(
            bg=colour
        )

        widgets = (
            self.trial_label,
            self.cue_label,
            self.countdown_label,
            self.progress_label,
            self.left_indicator,
            self.right_indicator,
        )

        for widget in widgets:
            widget.configure(
                bg=colour
            )

        self.side_frame.configure(
            bg=colour
        )

        self.timer_frame.configure(
            bg=colour
        )

        self.bottom_frame.configure(
            bg=colour
        )

        self.countdown_canvas.configure(
            bg=colour
        )

    def show_circle_timer(self) -> None:
        self.countdown_label.grid_remove()
        self.countdown_canvas.grid()

    def show_standard_timer(self) -> None:
        self.countdown_canvas.grid_remove()
        self.countdown_label.grid()

    def draw_prepare_circle(
        self,
        remaining: float,
    ) -> None:
        """Render circular camera/user position countdown."""

        self.countdown_canvas.delete(
            "all"
        )

        size = 190
        centre = size / 2.0
        padding = 18

        self.countdown_canvas.create_oval(
            padding,
            padding,
            size - padding,
            size - padding,
            outline=CIRCLE_TRACK,
            width=11,
        )

        fraction = max(
            0.0,
            min(
                1.0,
                remaining / PREPARE_TIME,
            ),
        )

        self.countdown_canvas.create_arc(
            padding,
            padding,
            size - padding,
            size - padding,
            start=90,
            extent=-360.0 * fraction,
            style="arc",
            outline=CIRCLE_ACTIVE,
            width=12,
        )

        seconds = max(
            0,
            math.ceil(remaining),
        )

        self.countdown_canvas.create_text(
            centre,
            centre - 5,
            text=str(seconds),
            fill=TEXT_MAIN,
            font=("Arial", 46, "bold"),
        )

        self.countdown_canvas.create_text(
            centre,
            centre + 38,
            text="SECONDS",
            fill="#bbbbbb",
            font=("Arial", 10, "bold"),
        )

    # ======================================================================
    # LSL MARKERS / LOCAL EVENT LOG
    # ======================================================================

    def emit_marker(
        self,
        marker: str,
        timestamp: float | None = None,
    ) -> float:
        """
        Publish LSL marker and mirror to local CSV event log.

        Return the timestamp used in event.
        """

        if timestamp is None:
            timestamp = local_clock()

        self.outlet.push_sample(
            [marker],
            timestamp=timestamp,
        )

        print(
            f"{timestamp:.6f} | {marker}"
        )

        if self.csv_writer is not None:
            trial = self.current_trial

            if trial is None:
                row = [
                    timestamp,
                    marker,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]

            else:
                row = [
                    timestamp,
                    marker,
                    trial.number,
                    trial.side,
                    trial.up_time,
                    trial.hold_time,
                    trial.down_time,
                    trial.rest_time,
                ]

            self.csv_writer.writerow(row)

            if self.csv_file is not None:
                self.csv_file.flush()

        return timestamp

    # ======================================================================
    # PHASE STATE MACHINE
    # ======================================================================

    def start_phase(
        self,
        phase: str,
        duration: float,
        background: str,
        marker: str | None = None,
    ) -> None:
        """
        Begin one timed experiment - phase start time and associated marker deliberately share same LSL clock timestamp.
        """

        if not self.running:
            return

        phase_timestamp = local_clock()

        self.phase_name = phase
        self.phase_duration = duration

        self.phase_start_clock = phase_timestamp
        self.phase_end_clock = (
            phase_timestamp + duration
        )

        self.set_background(
            background
        )

        if marker is not None:
            self.emit_marker(
                marker,
                timestamp=phase_timestamp,
            )

        self.update_phase_display()

    def update_phase_display(self) -> None:
        """Update countdown graphics and transition-complete phase/s."""

        if not self.running:
            return

        if self.phase_end_clock is None:
            return

        now = local_clock()

        remaining = max(
            0.0,
            self.phase_end_clock - now,
        )

        self.render_phase(
            remaining
        )

        if remaining <= 0:
            self.advance_phase()
            return

        self.root.after(
            UPDATE_INTERVAL_MS,
            self.update_phase_display,
        )

    def render_phase(
        self,
        remaining: float,
    ) -> None:
        """Render visual state associated with current phase."""

        if self.phase_name == "PREPARE":
            self.set_active_side(None)
            self.show_circle_timer()

            self.trial_label.configure(
                text="Camera positioning"
            )

            self.cue_label.configure(
                text=(
                    "THE TEST IS BEGINNING\n\n"
                    "PLEASE STAND IN FRONT\n"
                    "OF THE CAMERA"
                ),
                font=("Arial", 44, "bold"),
            )

            self.draw_prepare_circle(
                remaining
            )

            return

        self.show_standard_timer()

        self.countdown_label.configure(
            text=f"{remaining:.1f} s"
        )

        if self.phase_name == "BASELINE":
            self.set_active_side(None)

            self.trial_label.configure(
                text="Initial EEG baseline"
            )

            self.cue_label.configure(
                text="RELAX\nARMS DOWN",
                font=("Arial", 64, "bold"),
            )

            return

        trial = self.current_trial

        if trial is None:
            return

        if self.phase_name == "REST":
            self.set_active_side(None)

            self.trial_label.configure(
                text=(
                    f"Trial {trial.number} / "
                    f"{len(self.trials)} complete"
                )
            )

            self.cue_label.configure(
                text="REST\nARMS DOWN",
                font=("Arial", 64, "bold"),
            )

            return

        self.set_active_side(
            trial.side
        )

        self.trial_label.configure(
            text=(
                f"Trial {trial.number} / "
                f"{len(self.trials)}"
            )
        )

        if self.phase_name == "UP":
            cue = f"{trial.side} ARM\n↑  UP"
            font_size = 70

        elif self.phase_name == "HOLD":
            cue = f"HOLD\n{trial.side} ARM"
            font_size = 66

        elif self.phase_name == "DOWN":
            cue = f"{trial.side} ARM\n↓  DOWN"
            font_size = 70

        else:
            return

        self.cue_label.configure(
            text=cue,
            font=("Arial", font_size, "bold"),
        )

    def advance_phase(self) -> None:
        """Advance experiment state machine."""

        if not self.running:
            return

        # PREPARE -> BASELINE
        if self.phase_name == "PREPARE":
            self.emit_marker(
                "PREPARE_END"
            )

            self.start_phase(
                phase="BASELINE",
                duration=INITIAL_BASELINE_TIME,
                background=BG_READY,
                marker="BASELINE_START",
            )

            return

        # BASELINE -> first trial
        if self.phase_name == "BASELINE":
            self.emit_marker(
                "BASELINE_END"
            )

            self.start_next_trial()
            return

        trial = self.current_trial

        if trial is None:
            return

        # UP -> HOLD
        if self.phase_name == "UP":
            self.start_phase(
                phase="HOLD",
                duration=trial.hold_time,
                background=BG_HOLD,
                marker=(
                    f"TRIAL_{trial.number:03d}_HOLD_START"
                ),
            )

            return

        # HOLD -> DOWN
        if self.phase_name == "HOLD":
            self.start_phase(
                phase="DOWN",
                duration=trial.down_time,
                background=(
                    BG_LEFT
                    if trial.side == "LEFT"
                    else BG_RIGHT
                ),
                marker=(
                    f"TRIAL_{trial.number:03d}_DOWN_START"
                ),
            )

            return

        # DOWN -> REST
        if self.phase_name == "DOWN":
            self.start_phase(
                phase="REST",
                duration=trial.rest_time,
                background=BG_REST,
                marker=(
                    f"TRIAL_{trial.number:03d}_REST_START"
                ),
            )

            return

        # REST -> next trial
        if self.phase_name == "REST":
            self.emit_marker(
                f"TRIAL_{trial.number:03d}_COMPLETE"
            )

            self.progress["value"] = trial.number

            self.progress_label.configure(
                text=(
                    f"{trial.number} / "
                    f"{len(self.trials)}"
                )
            )

            self.start_next_trial()

    # ======================================================================
    # TRIAL MANAGEMENT
    # ======================================================================

    def start_next_trial(self) -> None:
        """Load the next trial and begin UP cue."""

        self.current_trial_index += 1

        if self.current_trial_index >= len(self.trials):
            self.finish_experiment()
            return

        self.current_trial = self.trials[
            self.current_trial_index
        ]

        trial = self.current_trial

        config_marker = (
            f"TRIAL_{trial.number:03d}_CONFIG"
            f"|SIDE={trial.side}"
            f"|UP={trial.up_time:.3f}"
            f"|HOLD={trial.hold_time:.3f}"
            f"|DOWN={trial.down_time:.3f}"
            f"|REST={trial.rest_time:.3f}"
        )

        self.emit_marker(
            config_marker
        )

        cue_marker = (
            f"TRIAL_{trial.number:03d}_CUE_{trial.side}"
        )

        self.start_phase(
            phase="UP",
            duration=trial.up_time,
            background=(
                BG_LEFT
                if trial.side == "LEFT"
                else BG_RIGHT
            ),
            marker=cue_marker,
        )

    # ======================================================================
    # FINISH / ABORT
    # ======================================================================

    def finish_experiment(self) -> None:
        """Complete experiment."""

        self.emit_marker(
            "EXPERIMENT_COMPLETE"
        )

        self.running = False

        self.close_log()

        self.set_background(
            BG_COMPLETE
        )

        self.set_active_side(None)
        self.show_standard_timer()

        self.trial_label.configure(
            text="Experiment complete"
        )

        self.cue_label.configure(
            text="COMPLETE\n✓",
            font=("Arial", 72, "bold"),
        )

        self.countdown_label.configure(
            text="You can now stop LabRecorder."
        )

        self.progress["value"] = len(
            self.trials
        )

        self.progress_label.configure(
            text=(
                f"{len(self.trials)} / "
                f"{len(self.trials)}"
            )
        )

    def abort_experiment(
        self,
        event=None,
    ) -> None:
        """Abort active experiment after confirmation."""

        if not self.running:
            return

        answer = messagebox.askyesno(
            "Abort experiment",
            "Stop the experiment?",
        )

        if not answer:
            return

        self.emit_marker(
            "EXPERIMENT_ABORTED"
        )

        self.running = False
        self.close_log()

        try:
            self.root.state("normal")
        except tk.TclError:
            pass

        self.build_setup_screen()

    def request_close(self) -> None:
        """Handle user-closing of app window."""

        if self.running:
            answer = messagebox.askyesno(
                "Close experiment",
                (
                    "An experiment is currently running.\n\n"
                    "Abort it and close the application?"
                ),
            )

            if not answer:
                return

            self.emit_marker(
                "EXPERIMENT_ABORTED"
            )

            self.running = False

        self.close_log()
        self.root.destroy()


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse optional runtime config"""

    parser = argparse.ArgumentParser(
        description=(
            "Run the ATS randomised LEFT / RIGHT motor-response experiment."
        )
    )

    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory for local CSV experiment logs. "
            "Overrides ATS_DATA_ROOT."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()

    log_directory = resolve_log_directory(
        args.log_dir
    )

    outlet = create_experiment_outlet()

    print()
    print("========================================")
    print(" ATS MOTOR EXPERIMENT")
    print(f" LSL stream: {LSL_STREAM_NAME}")
    print(f" Log dir:    {log_directory}")
    print("========================================")
    print()

    root = tk.Tk()

    MotorExperimentApp(
        root=root,
        outlet=outlet,
        log_directory=log_directory,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
