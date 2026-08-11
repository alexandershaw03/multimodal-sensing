import csv
import math
import random
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from pylsl import StreamInfo, StreamOutlet, local_clock


# =========================================================
# ATS AUTOMATIC MOTOR EXPERIMENT V2
#
# Hands-free LEFT / RIGHT motor experiment
#
# Trial:
#
#     ARM UP
#       ↓
#     HOLD
#       ↓
#     ARM DOWN
#       ↓
#     REST
#
# Features:
#
#   - balanced random LEFT / RIGHT sequence
#   - random HOLD duration
#   - random REST duration
#   - 10 s camera positioning countdown
#   - separate EEG baseline
#   - large side indicators
#   - progress bar
#   - automatic LSL markers
#   - local CSV backup log
#
# LSL stream:
#     ATS_EXPERIMENT
#
# =========================================================


# =========================================================
# DEFAULT SETTINGS
# =========================================================

DEFAULT_TRIALS_PER_ARM = 10

DEFAULT_TARGET_MINUTES = 8.0


# Movement timing
DEFAULT_UP_TIME = 1.5
DEFAULT_DOWN_TIME = 1.5

DEFAULT_HOLD_MIN = 2.0
DEFAULT_HOLD_MAX = 6.0

DEFAULT_REST_MIN = 4.0
DEFAULT_REST_MAX = 7.0


# Before experiment
PREPARE_TIME = 10.0
INITIAL_BASELINE_TIME = 5.0


# GUI refresh
UPDATE_INTERVAL_MS = 50


# =========================================================
# COLOURS
# =========================================================

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


# =========================================================
# LSL STREAM
# =========================================================

experiment_info = StreamInfo(
    name="ATS_EXPERIMENT",
    type="Markers",
    channel_count=1,
    nominal_srate=0,
    channel_format="string",
    source_id="ats_motor_experiment_v2"
)


experiment_info.desc().append_child_value(
    "manufacturer",
    "ATS"
)

experiment_info.desc().append_child_value(
    "protocol",
    "Randomised Left Right Motor Experiment"
)

experiment_info.desc().append_child_value(
    "version",
    "2.0"
)


experiment_outlet = StreamOutlet(
    experiment_info
)


# =========================================================
# HELPERS
# =========================================================

def random_balanced_sequence(trials_per_arm):
    """
    Create a balanced LEFT / RIGHT sequence.

    Prevents more than two identical sides appearing
    consecutively wherever possible.
    """

    remaining = {
        "LEFT": trials_per_arm,
        "RIGHT": trials_per_arm
    }

    sequence = []


    while (
        remaining["LEFT"] > 0
        or
        remaining["RIGHT"] > 0
    ):

        choices = [
            side
            for side in ["LEFT", "RIGHT"]
            if remaining[side] > 0
        ]


        # Prevent three identical trials in a row.
        if (
            len(sequence) >= 2
            and
            sequence[-1] == sequence[-2]
        ):

            repeated_side = sequence[-1]

            filtered = [
                side
                for side in choices
                if side != repeated_side
            ]

            if filtered:

                choices = filtered


        side = random.choice(
            choices
        )


        sequence.append(
            side
        )

        remaining[side] -= 1


    return sequence


def estimate_trials_per_arm(
    target_minutes,
    up_time,
    down_time,
    hold_min,
    hold_max,
    rest_min,
    rest_max
):
    """
    Estimate how many balanced trials fit approximately
    within a requested experiment duration.
    """

    mean_hold = (
        hold_min + hold_max
    ) / 2.0


    mean_rest = (
        rest_min + rest_max
    ) / 2.0


    mean_trial_duration = (
        up_time
        +
        mean_hold
        +
        down_time
        +
        mean_rest
    )


    target_seconds = (
        target_minutes
        *
        60.0
    )


    usable_seconds = max(
        1.0,
        target_seconds
        -
        PREPARE_TIME
        -
        INITIAL_BASELINE_TIME
    )


    estimated_total_trials = max(
        2,
        int(
            round(
                usable_seconds
                /
                mean_trial_duration
            )
        )
    )


    # Keep total even for equal LEFT / RIGHT.
    if estimated_total_trials % 2 != 0:

        estimated_total_trials -= 1


    estimated_total_trials = max(
        2,
        estimated_total_trials
    )


    return (
        estimated_total_trials // 2
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

class ATSExperiment:

    def __init__(
        self,
        root
    ):

        self.root = root


        self.root.title(
            "ATS Motor Experiment"
        )


        self.root.configure(
            bg=BG_DEFAULT
        )


        self.root.geometry(
            "1000x760"
        )


        # =================================================
        # RUNTIME STATE
        # =================================================

        self.running = False


        self.trials = []

        self.current_trial_index = -1

        self.current_trial = None


        self.phase_name = None

        self.phase_start_clock = None

        self.phase_end_clock = None

        self.phase_duration = None


        self.csv_file = None

        self.csv_writer = None


        # Experiment timing settings
        self.up_time = None
        self.down_time = None

        self.hold_min = None
        self.hold_max = None

        self.rest_min = None
        self.rest_max = None


        # Emergency abort
        self.root.bind(
            "<Escape>",
            self.abort_experiment
        )


        self.build_setup_screen()


    # =====================================================
    # WINDOW UTILITIES
    # =====================================================

    def clear_window(self):

        for widget in self.root.winfo_children():

            widget.destroy()


    # =====================================================
    # SETUP SCREEN
    # =====================================================

    def build_setup_screen(self):

        self.clear_window()

        self.running = False


        self.root.configure(
            bg=BG_DEFAULT
        )


        # ---------------------------------------------
        # TITLE
        # ---------------------------------------------

        title = tk.Label(
            self.root,
            text="ATS MOTOR EXPERIMENT",
            font=("Arial", 30, "bold"),
            fg="white",
            bg=BG_DEFAULT
        )


        title.pack(
            pady=(30, 10)
        )


        subtitle = tk.Label(
            self.root,
            text=(
                "Hands-free randomised LEFT / RIGHT motor test"
            ),
            font=("Arial", 14),
            fg="#cccccc",
            bg=BG_DEFAULT
        )


        subtitle.pack(
            pady=(0, 25)
        )


        # ---------------------------------------------
        # SETTINGS FRAME
        # ---------------------------------------------

        frame = tk.Frame(
            self.root,
            bg=BG_DEFAULT
        )


        frame.pack(
            padx=40,
            pady=10
        )


        # =============================================
        # EXPERIMENT LENGTH
        # =============================================

        self.mode_var = tk.StringVar(
            value="trials"
        )


        tk.Label(
            frame,
            text="Experiment length:",
            font=("Arial", 13, "bold"),
            fg="white",
            bg=BG_DEFAULT
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=8
        )


        tk.Radiobutton(
            frame,
            text="Trials per arm",
            variable=self.mode_var,
            value="trials",
            fg="white",
            bg=BG_DEFAULT,
            selectcolor=BG_DEFAULT,
            activebackground=BG_DEFAULT,
            activeforeground="white"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )


        self.trials_entry = tk.Entry(
            frame,
            width=10
        )


        self.trials_entry.insert(
            0,
            str(DEFAULT_TRIALS_PER_ARM)
        )


        self.trials_entry.grid(
            row=1,
            column=1,
            padx=10
        )


        tk.Radiobutton(
            frame,
            text="Approximate duration (minutes)",
            variable=self.mode_var,
            value="duration",
            fg="white",
            bg=BG_DEFAULT,
            selectcolor=BG_DEFAULT,
            activebackground=BG_DEFAULT,
            activeforeground="white"
        ).grid(
            row=2,
            column=0,
            sticky="w"
        )


        self.duration_entry = tk.Entry(
            frame,
            width=10
        )


        self.duration_entry.insert(
            0,
            str(DEFAULT_TARGET_MINUTES)
        )


        self.duration_entry.grid(
            row=2,
            column=1,
            padx=10
        )


        # =============================================
        # MOVEMENT TIMING
        # =============================================

        tk.Label(
            frame,
            text="Movement timing:",
            font=("Arial", 13, "bold"),
            fg="white",
            bg=BG_DEFAULT
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(25, 8)
        )


        labels = [

            (
                "UP cue duration (s)",
                DEFAULT_UP_TIME
            ),

            (
                "DOWN cue duration (s)",
                DEFAULT_DOWN_TIME
            ),

            (
                "Minimum HOLD time (s)",
                DEFAULT_HOLD_MIN
            ),

            (
                "Maximum HOLD time (s)",
                DEFAULT_HOLD_MAX
            ),

            (
                "Minimum REST time (s)",
                DEFAULT_REST_MIN
            ),

            (
                "Maximum REST time (s)",
                DEFAULT_REST_MAX
            )

        ]


        self.timing_entries = []


        for i, (
            label_text,
            default_value
        ) in enumerate(
            labels,
            start=4
        ):

            tk.Label(
                frame,
                text=label_text,
                font=("Arial", 11),
                fg="#dddddd",
                bg=BG_DEFAULT
            ).grid(
                row=i,
                column=0,
                sticky="w",
                pady=3
            )


            entry = tk.Entry(
                frame,
                width=10
            )


            entry.insert(
                0,
                str(default_value)
            )


            entry.grid(
                row=i,
                column=1,
                padx=10,
                pady=3
            )


            self.timing_entries.append(
                entry
            )


        # =============================================
        # START BUTTON
        # =============================================

        start_button = tk.Button(
            self.root,
            text="START EXPERIMENT",
            font=("Arial", 16, "bold"),
            padx=30,
            pady=12,
            command=self.prepare_experiment
        )


        start_button.pack(
            pady=30
        )


        info = tk.Label(
            self.root,
            text=(
                "The LEFT / RIGHT order, HOLD duration and REST duration "
                "are randomised.\n"
                "A 10-second camera-positioning countdown occurs before "
                "the baseline begins.\n"
                "No keyboard input is required once the test starts.\n"
                "Press ESC only if you need to abort."
            ),
            font=("Arial", 11),
            fg="#aaaaaa",
            bg=BG_DEFAULT,
            justify="center"
        )


        info.pack()


    # =====================================================
    # PREPARE EXPERIMENT
    # =====================================================

    def prepare_experiment(self):

        try:

            self.up_time = float(
                self.timing_entries[0].get()
            )


            self.down_time = float(
                self.timing_entries[1].get()
            )


            self.hold_min = float(
                self.timing_entries[2].get()
            )


            self.hold_max = float(
                self.timing_entries[3].get()
            )


            self.rest_min = float(
                self.timing_entries[4].get()
            )


            self.rest_max = float(
                self.timing_entries[5].get()
            )


            # -----------------------------------------
            # BASIC VALIDATION
            # -----------------------------------------

            if (
                self.up_time <= 0
                or
                self.down_time <= 0
                or
                self.hold_min <= 0
                or
                self.hold_max <= 0
                or
                self.rest_min <= 0
                or
                self.rest_max <= 0
            ):

                raise ValueError


            if (
                self.hold_max
                <
                self.hold_min
            ):

                raise ValueError


            if (
                self.rest_max
                <
                self.rest_min
            ):

                raise ValueError


            # -----------------------------------------
            # TRIAL MODE
            # -----------------------------------------

            if (
                self.mode_var.get()
                ==
                "trials"
            ):

                trials_per_arm = int(
                    self.trials_entry.get()
                )


                if trials_per_arm < 1:

                    raise ValueError


            # -----------------------------------------
            # DURATION MODE
            # -----------------------------------------

            else:

                target_minutes = float(
                    self.duration_entry.get()
                )


                if target_minutes <= 0:

                    raise ValueError


                trials_per_arm = (
                    estimate_trials_per_arm(
                        target_minutes,
                        self.up_time,
                        self.down_time,
                        self.hold_min,
                        self.hold_max,
                        self.rest_min,
                        self.rest_max
                    )
                )


        except ValueError:

            messagebox.showerror(
                "Invalid settings",
                "Please check the experiment settings."
            )

            return


        # =============================================
        # CREATE RANDOM BALANCED SEQUENCE
        # =============================================

        sequence = random_balanced_sequence(
            trials_per_arm
        )


        self.trials = []


        for number, side in enumerate(
            sequence,
            start=1
        ):

            hold_time = random.uniform(
                self.hold_min,
                self.hold_max
            )


            rest_time = random.uniform(
                self.rest_min,
                self.rest_max
            )


            self.trials.append(
                {
                    "trial": number,

                    "side": side,

                    "up_time":
                        self.up_time,

                    "hold_time":
                        hold_time,

                    "down_time":
                        self.down_time,

                    "rest_time":
                        rest_time
                }
            )


        # =============================================
        # LOG + GUI
        # =============================================

        self.create_log_file()

        self.build_experiment_screen()


        self.running = True

        self.current_trial_index = -1

        self.current_trial = None


        # =============================================
        # START
        # =============================================

        self.emit_marker(
            "EXPERIMENT_START"
        )


        self.start_phase(
            phase="PREPARE",
            duration=PREPARE_TIME,
            background=BG_READY,
            marker="PREPARE_START"
        )


    # =====================================================
    # CSV LOG
    # =====================================================

    def create_log_file(self):

        log_dir = (
            Path.home()
            /
            "Documents"
            /
            "ATS_Multimodal"
            /
            "experiment_logs"
        )


        log_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )


        log_path = (
            log_dir
            /
            (
                "ATS_motor_experiment_v2_"
                +
                timestamp
                +
                ".csv"
            )
        )


        self.csv_file = open(
            log_path,
            "w",
            newline="",
            encoding="utf-8"
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
                "planned_rest_s"
            ]
        )


        self.csv_file.flush()


        print()
        print(
            "Experiment log:"
        )

        print(
            log_path
        )

        print()


    # =====================================================
    # BUILD EXPERIMENT SCREEN
    # =====================================================

    def build_experiment_screen(self):

        self.clear_window()


        self.root.configure(
            bg=BG_READY
        )


        # Windows maximised mode
        try:

            self.root.state(
                "zoomed"
            )

        except tk.TclError:

            pass


        # =============================================
        # TOP STATUS
        # =============================================

        self.trial_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 20, "bold"),
            fg="white",
            bg=BG_READY
        )


        self.trial_label.pack(
            pady=(20, 5)
        )


        # =============================================
        # SIDE INDICATORS
        # =============================================

        self.side_frame = tk.Frame(
            self.root,
            bg=BG_READY
        )


        self.side_frame.pack(
            fill="x",
            padx=50,
            pady=(5, 0)
        )


        self.left_indicator = tk.Label(
            self.side_frame,
            text="◀  LEFT",
            font=("Arial", 28, "bold"),
            fg=TEXT_DIM,
            bg=BG_READY
        )


        self.left_indicator.pack(
            side="left",
            anchor="w"
        )


        self.right_indicator = tk.Label(
            self.side_frame,
            text="RIGHT  ▶",
            font=("Arial", 28, "bold"),
            fg=TEXT_DIM,
            bg=BG_READY
        )


        self.right_indicator.pack(
            side="right",
            anchor="e"
        )


        # =============================================
        # MAIN CUE
        # =============================================

        self.cue_label = tk.Label(
            self.root,
            text="GET READY",
            font=("Arial", 64, "bold"),
            fg="white",
            bg=BG_READY,
            justify="center"
        )


        self.cue_label.pack(
            expand=True
        )


        # =============================================
        # TIMER AREA
        # =============================================

        self.timer_frame = tk.Frame(
            self.root,
            bg=BG_READY
        )


        self.timer_frame.pack(
            pady=5
        )


        # ---------------------------------------------
        # Circular PREPARE timer
        # ---------------------------------------------

        self.countdown_canvas = tk.Canvas(
            self.timer_frame,
            width=190,
            height=190,
            bg=BG_READY,
            highlightthickness=0
        )


        self.countdown_canvas.grid(
            row=0,
            column=0
        )


        # ---------------------------------------------
        # Normal phase countdown
        # ---------------------------------------------

        self.countdown_label = tk.Label(
            self.timer_frame,
            text="",
            font=("Arial", 28, "bold"),
            fg="white",
            bg=BG_READY
        )


        self.countdown_label.grid(
            row=0,
            column=0
        )


        # Circle starts hidden until PREPARE
        self.countdown_canvas.grid_remove()


        # =============================================
        # PROGRESS
        # =============================================

        self.bottom_frame = tk.Frame(
            self.root,
            bg=BG_READY
        )


        self.bottom_frame.pack(
            fill="x",
            padx=80,
            pady=(10, 25)
        )


        self.progress = ttk.Progressbar(
            self.bottom_frame,
            orient="horizontal",
            mode="determinate",
            maximum=len(self.trials)
        )


        self.progress.pack(
            fill="x"
        )


        self.progress_label = tk.Label(
            self.bottom_frame,
            text=(
                "0 / {}".format(
                    len(self.trials)
                )
            ),
            font=("Arial", 14),
            fg="white",
            bg=BG_READY
        )


        self.progress_label.pack(
            pady=8
        )


    # =====================================================
    # SIDE INDICATORS
    # =====================================================

    def set_active_side(
        self,
        side=None
    ):

        if side == "LEFT":

            self.left_indicator.configure(
                fg=TEXT_LEFT
            )

            self.right_indicator.configure(
                fg=TEXT_DIM
            )


        elif side == "RIGHT":

            self.left_indicator.configure(
                fg=TEXT_DIM
            )

            self.right_indicator.configure(
                fg=TEXT_RIGHT
            )


        else:

            self.left_indicator.configure(
                fg=TEXT_DIM
            )

            self.right_indicator.configure(
                fg=TEXT_DIM
            )


    # =====================================================
    # CIRCULAR COUNTDOWN
    # =====================================================

    def draw_prepare_circle(
        self,
        remaining
    ):

        self.countdown_canvas.delete(
            "all"
        )


        size = 190

        centre = size / 2.0

        padding = 18


        # ---------------------------------------------
        # Background ring
        # ---------------------------------------------

        self.countdown_canvas.create_oval(
            padding,
            padding,
            size - padding,
            size - padding,
            outline=CIRCLE_TRACK,
            width=11
        )


        # ---------------------------------------------
        # Progress arc
        # ---------------------------------------------

        fraction = max(
            0.0,
            min(
                1.0,
                remaining
                /
                PREPARE_TIME
            )
        )


        self.countdown_canvas.create_arc(
            padding,
            padding,
            size - padding,
            size - padding,
            start=90,
            extent=(
                -360.0
                *
                fraction
            ),
            style="arc",
            outline=CIRCLE_ACTIVE,
            width=12
        )


        # ---------------------------------------------
        # Number
        # ---------------------------------------------

        seconds = max(
            0,
            int(
                math.ceil(
                    remaining
                )
            )
        )


        self.countdown_canvas.create_text(
            centre,
            centre - 5,
            text=str(seconds),
            fill="white",
            font=("Arial", 46, "bold")
        )


        self.countdown_canvas.create_text(
            centre,
            centre + 38,
            text="SECONDS",
            fill="#bbbbbb",
            font=("Arial", 10, "bold")
        )


    # =====================================================
    # TIMER DISPLAY MODE
    # =====================================================

    def show_circle_timer(self):

        self.countdown_label.grid_remove()

        self.countdown_canvas.grid()


    def show_standard_timer(self):

        self.countdown_canvas.grid_remove()

        self.countdown_label.grid()


    # =====================================================
    # LSL MARKERS
    # =====================================================

    def emit_marker(
        self,
        marker,
        timestamp=None
    ):

        if timestamp is None:

            timestamp = local_clock()


        experiment_outlet.push_sample(
            [marker],
            timestamp=timestamp
        )


        print(
            "{:.6f} | {}".format(
                timestamp,
                marker
            )
        )


        # =============================================
        # LOCAL CSV BACKUP
        # =============================================

        if (
            self.csv_writer
            is not None
        ):

            if (
                self.current_trial
                is None
            ):

                trial_number = ""
                side = ""

                up_time = ""
                hold_time = ""
                down_time = ""
                rest_time = ""


            else:

                trial_number = (
                    self.current_trial[
                        "trial"
                    ]
                )

                side = (
                    self.current_trial[
                        "side"
                    ]
                )

                up_time = (
                    self.current_trial[
                        "up_time"
                    ]
                )

                hold_time = (
                    self.current_trial[
                        "hold_time"
                    ]
                )

                down_time = (
                    self.current_trial[
                        "down_time"
                    ]
                )

                rest_time = (
                    self.current_trial[
                        "rest_time"
                    ]
                )


            self.csv_writer.writerow(
                [
                    timestamp,
                    marker,
                    trial_number,
                    side,
                    up_time,
                    hold_time,
                    down_time,
                    rest_time
                ]
            )


            self.csv_file.flush()


    # =====================================================
    # PHASE MANAGEMENT
    # =====================================================

    def start_phase(
        self,
        phase,
        duration,
        background,
        marker=None
    ):

        if not self.running:

            return


        self.phase_name = phase

        self.phase_duration = duration


        phase_timestamp = (
            local_clock()
        )


        self.phase_start_clock = (
            phase_timestamp
        )


        self.phase_end_clock = (
            phase_timestamp
            +
            duration
        )


        self.set_background(
            background
        )


        if marker:

            self.emit_marker(
                marker,
                timestamp=phase_timestamp
            )


        self.update_phase_display()


    # =====================================================
    # BACKGROUND
    # =====================================================

    def set_background(
        self,
        colour
    ):

        self.root.configure(
            bg=colour
        )


        widgets = [

            self.trial_label,

            self.cue_label,

            self.countdown_label,

            self.progress_label,

            self.left_indicator,

            self.right_indicator

        ]


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


    # =====================================================
    # UPDATE PHASE DISPLAY
    # =====================================================

    def update_phase_display(self):

        if not self.running:

            return


        now = local_clock()


        remaining = max(
            0.0,
            self.phase_end_clock
            -
            now
        )


        # =============================================
        # PREPARE
        # =============================================

        if self.phase_name == "PREPARE":

            self.set_active_side(
                None
            )


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
                font=("Arial", 44, "bold")
            )


            self.draw_prepare_circle(
                remaining
            )


        # =============================================
        # EEG BASELINE
        # =============================================

        elif self.phase_name == "BASELINE":

            self.set_active_side(
                None
            )


            self.show_standard_timer()


            self.trial_label.configure(
                text="Initial EEG baseline"
            )


            self.cue_label.configure(
                text=(
                    "RELAX\n"
                    "ARMS DOWN"
                ),
                font=("Arial", 64, "bold")
            )


            self.countdown_label.configure(
                text=(
                    "{:.1f} s".format(
                        remaining
                    )
                )
            )


        # =============================================
        # UP
        # =============================================

        elif self.phase_name == "UP":

            side = (
                self.current_trial[
                    "side"
                ]
            )


            self.set_active_side(
                side
            )


            self.show_standard_timer()


            self.trial_label.configure(
                text=(
                    "Trial {} / {}".format(
                        self.current_trial[
                            "trial"
                        ],
                        len(self.trials)
                    )
                )
            )


            self.cue_label.configure(
                text=(
                    "{} ARM\n"
                    "↑  UP"
                ).format(
                    side
                ),
                font=("Arial", 70, "bold")
            )


            self.countdown_label.configure(
                text=(
                    "{:.1f} s".format(
                        remaining
                    )
                )
            )


        # =============================================
        # HOLD
        # =============================================

        elif self.phase_name == "HOLD":

            side = (
                self.current_trial[
                    "side"
                ]
            )


            self.set_active_side(
                side
            )


            self.show_standard_timer()


            self.trial_label.configure(
                text=(
                    "Trial {} / {}".format(
                        self.current_trial[
                            "trial"
                        ],
                        len(self.trials)
                    )
                )
            )


            self.cue_label.configure(
                text=(
                    "HOLD\n"
                    "{} ARM"
                ).format(
                    side
                ),
                font=("Arial", 66, "bold")
            )


            self.countdown_label.configure(
                text=(
                    "{:.1f} s".format(
                        remaining
                    )
                )
            )


        # =============================================
        # DOWN
        # =============================================

        elif self.phase_name == "DOWN":

            side = (
                self.current_trial[
                    "side"
                ]
            )


            self.set_active_side(
                side
            )


            self.show_standard_timer()


            self.trial_label.configure(
                text=(
                    "Trial {} / {}".format(
                        self.current_trial[
                            "trial"
                        ],
                        len(self.trials)
                    )
                )
            )


            self.cue_label.configure(
                text=(
                    "{} ARM\n"
                    "↓  DOWN"
                ).format(
                    side
                ),
                font=("Arial", 70, "bold")
            )


            self.countdown_label.configure(
                text=(
                    "{:.1f} s".format(
                        remaining
                    )
                )
            )


        # =============================================
        # REST
        # =============================================

        elif self.phase_name == "REST":

            self.set_active_side(
                None
            )


            self.show_standard_timer()


            self.trial_label.configure(
                text=(
                    "Trial {} / {} complete".format(
                        self.current_trial[
                            "trial"
                        ],
                        len(self.trials)
                    )
                )
            )


            self.cue_label.configure(
                text=(
                    "REST\n"
                    "ARMS DOWN"
                ),
                font=("Arial", 64, "bold")
            )


            self.countdown_label.configure(
                text=(
                    "{:.1f} s".format(
                        remaining
                    )
                )
            )


        # =============================================
        # PHASE COMPLETE?
        # =============================================

        if remaining <= 0:

            self.advance_phase()

            return


        self.root.after(
            UPDATE_INTERVAL_MS,
            self.update_phase_display
        )


    # =====================================================
    # ADVANCE PHASE
    # =====================================================

    def advance_phase(self):

        if not self.running:

            return


        # =============================================
        # PREPARE -> BASELINE
        # =============================================

        if self.phase_name == "PREPARE":

            self.emit_marker(
                "PREPARE_END"
            )


            self.start_phase(
                phase="BASELINE",
                duration=INITIAL_BASELINE_TIME,
                background=BG_READY,
                marker="BASELINE_START"
            )


            return


        # =============================================
        # BASELINE -> FIRST TRIAL
        # =============================================

        if self.phase_name == "BASELINE":

            self.emit_marker(
                "BASELINE_END"
            )


            self.start_next_trial()


            return


        # =============================================
        # UP -> HOLD
        # =============================================

        if self.phase_name == "UP":

            trial = (
                self.current_trial
            )


            self.start_phase(
                phase="HOLD",
                duration=trial[
                    "hold_time"
                ],
                background=BG_HOLD,
                marker=(
                    "TRIAL_{:03d}_HOLD_START".format(
                        trial[
                            "trial"
                        ]
                    )
                )
            )


            return


        # =============================================
        # HOLD -> DOWN
        # =============================================

        if self.phase_name == "HOLD":

            trial = (
                self.current_trial
            )


            self.start_phase(
                phase="DOWN",
                duration=trial[
                    "down_time"
                ],
                background=(
                    BG_LEFT
                    if trial["side"] == "LEFT"
                    else BG_RIGHT
                ),
                marker=(
                    "TRIAL_{:03d}_DOWN_START".format(
                        trial[
                            "trial"
                        ]
                    )
                )
            )


            return


        # =============================================
        # DOWN -> REST
        # =============================================

        if self.phase_name == "DOWN":

            trial = (
                self.current_trial
            )


            self.start_phase(
                phase="REST",
                duration=trial[
                    "rest_time"
                ],
                background=BG_REST,
                marker=(
                    "TRIAL_{:03d}_REST_START".format(
                        trial[
                            "trial"
                        ]
                    )
                )
            )


            return


        # =============================================
        # REST -> NEXT TRIAL
        # =============================================

        if self.phase_name == "REST":

            trial = (
                self.current_trial
            )


            self.emit_marker(
                "TRIAL_{:03d}_COMPLETE".format(
                    trial[
                        "trial"
                    ]
                )
            )


            self.progress[
                "value"
            ] = trial[
                "trial"
            ]


            self.progress_label.configure(
                text=(
                    "{} / {}".format(
                        trial[
                            "trial"
                        ],
                        len(self.trials)
                    )
                )
            )


            self.start_next_trial()


    # =====================================================
    # START NEXT TRIAL
    # =====================================================

    def start_next_trial(self):

        self.current_trial_index += 1


        # =============================================
        # ALL DONE
        # =============================================

        if (
            self.current_trial_index
            >=
            len(self.trials)
        ):

            self.finish_experiment()

            return


        # =============================================
        # LOAD TRIAL
        # =============================================

        self.current_trial = (
            self.trials[
                self.current_trial_index
            ]
        )


        trial = (
            self.current_trial
        )


        side = (
            trial[
                "side"
            ]
        )


        # =============================================
        # CONFIG MARKER
        # =============================================

        config_marker = (

            "TRIAL_{:03d}_CONFIG"
            "|SIDE={}"
            "|UP={:.3f}"
            "|HOLD={:.3f}"
            "|DOWN={:.3f}"
            "|REST={:.3f}"

        ).format(

            trial[
                "trial"
            ],

            side,

            trial[
                "up_time"
            ],

            trial[
                "hold_time"
            ],

            trial[
                "down_time"
            ],

            trial[
                "rest_time"
            ]

        )


        self.emit_marker(
            config_marker
        )


        # =============================================
        # MAIN CUE
        # =============================================

        cue_marker = (

            "TRIAL_{:03d}_CUE_{}"

        ).format(

            trial[
                "trial"
            ],

            side

        )


        self.start_phase(
            phase="UP",

            duration=trial[
                "up_time"
            ],

            background=(
                BG_LEFT
                if side == "LEFT"
                else BG_RIGHT
            ),

            marker=cue_marker
        )


    # =====================================================
    # FINISH
    # =====================================================

    def finish_experiment(self):

        self.emit_marker(
            "EXPERIMENT_COMPLETE"
        )


        self.running = False


        if (
            self.csv_file
            is not None
        ):

            self.csv_file.close()

            self.csv_file = None


        self.set_background(
            BG_COMPLETE
        )


        self.set_active_side(
            None
        )


        self.show_standard_timer()


        self.trial_label.configure(
            text="Experiment complete"
        )


        self.cue_label.configure(
            text=(
                "COMPLETE\n"
                "✓"
            ),
            font=("Arial", 72, "bold")
        )


        self.countdown_label.configure(
            text=(
                "You can stop LabRecorder."
            )
        )


        self.progress[
            "value"
        ] = len(
            self.trials
        )


        self.progress_label.configure(
            text=(
                "{} / {}".format(
                    len(self.trials),
                    len(self.trials)
                )
            )
        )


    # =====================================================
    # ABORT
    # =====================================================

    def abort_experiment(
        self,
        event=None
    ):

        if not self.running:

            return


        answer = messagebox.askyesno(
            "Abort experiment",
            "Stop the experiment?"
        )


        if not answer:

            return


        self.emit_marker(
            "EXPERIMENT_ABORTED"
        )


        self.running = False


        if (
            self.csv_file
            is not None
        ):

            self.csv_file.close()

            self.csv_file = None


        try:

            self.root.state(
                "normal"
            )

        except tk.TclError:

            pass


        self.build_setup_screen()


# =========================================================
# MAIN
# =========================================================

def main():

    root = tk.Tk()


    app = ATSExperiment(
        root
    )


    root.mainloop()


if __name__ == "__main__":

    main()
