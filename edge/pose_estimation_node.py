"""
Jetson edge pose-estimation and movement-event node.

Runs NVIDIA PoseNet on a Jetson Nano camera stream, derives upper-limb kinematics, detects arm movement and publishes two LSL streams for wider multimodal-sensing system.

Published streams
-----------------
ATS_BODY_POSE
32-channel, irregular-rate float stream, containing:
    - smoothed camera-space joints
    - elbow angles
    - body-normalised wrist positions and velocities
    - rolling movement features
    - PoseNet FPS
    - trial-state diagnostics

ATS_VISION_EVENTS
    Irregular string marker stream containing:
    - raw LEFT/RIGHT movement
    - start/stop events
    - higher-level trial-state events

Timing
------
The Jetson camera API does not expose a sensor-hardware timestamp to this script.  
Each captured frame is, therefore, assigned one software-side LSL-clock timestamp immediately after Capture() returns.  
The pose sample and any vision events derived from that frame share that timestamp.
Internal elapsed-time calculations use time.monotonic().

This file intentionally remains compatible with Python 3.6, (as used by the Jetson Nano)

Usage
--------
    python edge/pose_estimation_node.py

    python edge/pose_estimation_node.py --camera csi://0 --network resnet18-body

    python edge/pose_estimation_node.py --headless
"""

import argparse
import math
import time
from collections import deque

import jetson_inference
import jetson_utils
from pylsl import StreamInfo, StreamOutlet, local_clock


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_NETWORK = "resnet18-body"
DEFAULT_CAMERA = "csi://0"
DEFAULT_THRESHOLD = 0.15
DEFAULT_PRINT_RATE_HZ = 5.0

POSITION_ALPHA = 0.35
VELOCITY_ALPHA = 0.30

# Raw-movement detector: speed + rolling displacement with hysteresis
MOVEMENT_START_SPEED = 0.38
MOVEMENT_STOP_SPEED = 0.18
MOTION_WINDOW = 0.50
WINDOW_DISPLACEMENT_THRESHOLD = 0.10
WINDOW_STOP_THRESHOLD = 0.035
STOP_DELAY = 0.30

# Reject small body-scales before body-normalised calculations
MIN_SHOULDER_WIDTH = 20.0

# Higher-level arm-trial state machine
NEUTRAL_CALIBRATION_TIME = 2.0
NEUTRAL_CALIBRATION_MAX_SPEED = 0.15
TRIAL_DEPARTURE_DISTANCE = 0.18
TRIAL_RETURN_DISTANCE = 0.10
TRIAL_RETURN_STABLE_TIME = 0.50
MIN_TRIAL_PEAK_DISTANCE = 0.22
MIN_TRIAL_DURATION = 0.50
REARM_STABLE_TIME = 1.00
NEUTRAL_ADAPT_ALPHA = 0.01
NEUTRAL_ADAPT_MAX_SPEED = 0.10

# Long tracking losses clear temporal state. Therefore, stale coordinates cannot create false velocity/movement events as user re-enters camera view
TRACKING_RESET_TIME = 0.75

POSE_STREAM_NAME = "ATS_BODY_POSE"
POSE_STREAM_TYPE = "Pose"
POSE_SOURCE_ID = "ats_jetson_body_pose_v2"

EVENT_STREAM_NAME = "ATS_VISION_EVENTS"
EVENT_STREAM_TYPE = "Markers"
EVENT_SOURCE_ID = "ats_jetson_vision_events_v2"

POSE_SOFTWARE_VERSION = "ATS_BODY_POSE_V2"
EVENT_SOFTWARE_VERSION = "ATS_VISION_EVENTS_V2"


# ============================================================================
# LSL BODY-POSE SCHEMA
# ============================================================================

POSE_CHANNELS = [
    # Camera-space joint coordinates
    ("L_SHOULDER_X", "pixels"),
    ("L_SHOULDER_Y", "pixels"),
    ("L_ELBOW_X", "pixels"),
    ("L_ELBOW_Y", "pixels"),
    ("L_WRIST_X", "pixels"),
    ("L_WRIST_Y", "pixels"),
    ("R_SHOULDER_X", "pixels"),
    ("R_SHOULDER_Y", "pixels"),
    ("R_ELBOW_X", "pixels"),
    ("R_ELBOW_Y", "pixels"),
    ("R_WRIST_X", "pixels"),
    ("R_WRIST_Y", "pixels"),

    # Elbow angles
    ("L_ELBOW_ANGLE", "degrees"),
    ("R_ELBOW_ANGLE", "degrees"),

    # Wrist coordinates relative to same-side shoulder (normalised by shoulder width)
    ("L_WRIST_REL_X", "shoulder_widths"),
    ("L_WRIST_REL_Y", "shoulder_widths"),
    ("R_WRIST_REL_X", "shoulder_widths"),
    ("R_WRIST_REL_Y", "shoulder_widths"),

    # Smoothed wrist speed
    ("L_WRIST_SPEED", "shoulder_widths_per_second"),
    ("R_WRIST_SPEED", "shoulder_widths_per_second"),

    # Rolling displacement over MOTION_WINDOW
    ("L_WINDOW_TRAVEL", "shoulder_widths"),
    ("R_WINDOW_TRAVEL", "shoulder_widths"),

    # Body scale/runtime performance
    ("SHOULDER_WIDTH", "pixels"),
    ("POSENET_FPS", "frames_per_second"),

    # Higher-level trial features
    ("L_NEUTRAL_DISTANCE", "shoulder_widths"),
    ("R_NEUTRAL_DISTANCE", "shoulder_widths"),
    ("L_TRIAL_PEAK_DISTANCE", "shoulder_widths"),
    ("R_TRIAL_PEAK_DISTANCE", "shoulder_widths"),
    ("L_TRIAL_ACTIVE", "boolean"),
    ("R_TRIAL_ACTIVE", "boolean"),
    ("L_TRIAL_READY", "boolean"),
    ("R_TRIAL_READY", "boolean"),
]

POSE_CHANNEL_INDEX = dict(
    (name, index)
    for index, (name, _unit) in enumerate(POSE_CHANNELS)
)


# ============================================================================
# NUMERICAL HELPERS
# ============================================================================


def get_keypoint(pose, wanted_id):
    """Return (x, y) for one PoseNet keypoint, or none if unavailable."""
    for keypoint in pose.Keypoints:
        if keypoint.ID == wanted_id:
            return (float(keypoint.x), float(keypoint.y))
    return None


def smooth_point(previous, current, alpha):
    """Exponential moving average, for 2D point."""
    if current is None:
        return previous
    if previous is None:
        return current

    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def smooth_value(previous, current, alpha):
    """Exponential moving average, for scalar."""
    if current is None:
        return previous
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def point_distance(a, b):
    """Euclidean distance between two 2D points."""
    if a is None or b is None:
        return None

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def vector_distance(a, b):
    """Euclidean distance between (two) body-relative 2D vectors"""
    return point_distance(a, b)


def angle_at_joint(a, b, c):
    """Return angle ABC in degrees, or none if geometry unavailable."""
    if a is None or b is None or c is None:
        return None

    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.sqrt(ba_x * ba_x + ba_y * ba_y)
    mag_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)

    if mag_ba == 0.0 or mag_bc == 0.0:
        return None

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def normalized_velocity(previous_point, current_point, dt, body_scale):
    """Return wrist speed in shoulder-widths per second"""
    if previous_point is None or current_point is None:
        return None
    if dt is None or dt <= 0.0:
        return None
    if body_scale is None or body_scale < MIN_SHOULDER_WIDTH:
        return None

    displacement = point_distance(previous_point, current_point)
    return (displacement / dt) / body_scale


def relative_point(point, reference, scale):
    """Return point, relative to reference, normalised by body scale"""
    if point is None or reference is None:
        return None
    if scale is None or scale < MIN_SHOULDER_WIDTH:
        return None

    return (
        (point[0] - reference[0]) / scale,
        (point[1] - reference[1]) / scale,
    )


def get_window_displacement(history, current_time):
    """Return wrist displacement, across approx MOTION_WINDOW seconds"""
    if len(history) < 2:
        return None

    current_point = history[-1][1]
    target_time = current_time - MOTION_WINDOW
    old_point = None

    for timestamp, point in history:
        if timestamp >= target_time:
            old_point = point
            break

    if old_point is None:
        old_point = history[0][1]

    return vector_distance(old_point, current_point)


def safe_value(value):
    """Convert unavailable numerical values to NaN for LSL output"""
    if value is None:
        return float("nan")
    return float(value)


def point_x(point):
    if point is None:
        return float("nan")
    return float(point[0])


def point_y(point):
    if point is None:
        return float("nan")
    return float(point[1])


def fmt(value):
    if value is None:
        return "N/A"
    return "{:.3f}".format(value)


# ============================================================================
# LSL HELPERS
# ============================================================================


def create_pose_outlet(network, camera_uri):
    """Create ATS_BODY_POSE LSL outlet and channel metadata"""
    info = StreamInfo(
        name=POSE_STREAM_NAME,
        type=POSE_STREAM_TYPE,
        channel_count=len(POSE_CHANNELS),
        nominal_srate=0,
        channel_format="float32",
        source_id=POSE_SOURCE_ID,
    )

    channels_xml = info.desc().append_child("channels")

    for label, unit in POSE_CHANNELS:
        channel = channels_xml.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        channel.append_child_value("type", "Pose")

    info.desc().append_child_value("manufacturer", "ATS")
    info.desc().append_child_value("device", "NVIDIA Jetson Nano")
    info.desc().append_child_value("vision_model", network)
    info.desc().append_child_value("camera", camera_uri)
    info.desc().append_child_value("software_version", POSE_SOFTWARE_VERSION)
    info.desc().append_child_value(
        "timestamp_semantics",
        "software LSL timestamp assigned immediately after camera capture returns",
    )

    return StreamOutlet(info)


def create_event_outlet():
    """Creates ATS_VISION_EVENTS LSL outlet"""
    info = StreamInfo(
        name=EVENT_STREAM_NAME,
        type=EVENT_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=EVENT_SOURCE_ID,
    )

    info.desc().append_child_value("manufacturer", "ATS")
    info.desc().append_child_value("device", "NVIDIA Jetson Nano")
    info.desc().append_child_value("software_version", EVENT_SOFTWARE_VERSION)
    info.desc().append_child_value(
        "timestamp_semantics",
        "derived event shares source camera-frame software LSL timestamp",
    )

    return StreamOutlet(info)


def emit_event(event_outlet, name, timestamp):
    """Print and publish one timestamped vision event"""
    print()
    print(">>> {}".format(name))
    event_outlet.push_sample([name], timestamp=timestamp)


# ============================================================================
# RAW MOVEMENT DETECTOR
# ============================================================================


class MovementDetector(object):
    """Per-arm movement detector, with start/stop hysteresis"""

    def __init__(self, side, event_outlet):
        self.side = side
        self.event_outlet = event_outlet
        self.moving = False
        self.last_motion_time = 0.0

    def reset(self):
        self.moving = False
        self.last_motion_time = 0.0

    def update(self, speed, window_displacement, monotonic_time, lsl_timestamp):
        if speed is None:
            return

        fast_motion = speed >= MOVEMENT_START_SPEED
        slow_motion = (
            window_displacement is not None
            and window_displacement >= WINDOW_DISPLACEMENT_THRESHOLD
        )

        if not self.moving:
            if fast_motion or slow_motion:
                self.moving = True
                self.last_motion_time = monotonic_time
                emit_event(
                    self.event_outlet,
                    "{}_MOVEMENT_START".format(self.side),
                    lsl_timestamp,
                )
            return

        still_moving = (
            speed >= MOVEMENT_STOP_SPEED
            or (
                window_displacement is not None
                and window_displacement >= WINDOW_STOP_THRESHOLD
            )
        )

        if still_moving:
            self.last_motion_time = monotonic_time
            return

        if monotonic_time - self.last_motion_time >= STOP_DELAY:
            self.moving = False
            emit_event(
                self.event_outlet,
                "{}_MOVEMENT_STOP".format(self.side),
                lsl_timestamp,
            )


# ============================================================================
# HIGHER-LEVEL ARM TRIAL STATE
# ============================================================================


class ArmTrialState(object):
    """Track one arm; from neutral calibration to complete movement trials"""

    def __init__(self, side, event_outlet):
        self.side = side
        self.event_outlet = event_outlet

        self.neutral_position = None
        self.calibration_start = None
        self.calibration_samples = []

        self.state = "UNCALIBRATED"
        self.trial_number = 0
        self.trial_start_time = None
        self.return_enter_time = None
        self.rearm_enter_time = None

        self.peak_distance = 0.0
        self.peak_emitted = False

    def reset_calibration(self):
        """Clear calibration and active trial state, after long tracking-loss"""
        self.neutral_position = None
        self.calibration_start = None
        self.calibration_samples = []
        self.state = "UNCALIBRATED"
        self.trial_start_time = None
        self.return_enter_time = None
        self.rearm_enter_time = None
        self.peak_distance = 0.0
        self.peak_emitted = False

    def update_calibration(self, current_position, speed, current_time, lsl_timestamp):
        if current_position is None or speed is None:
            return

        if speed > NEUTRAL_CALIBRATION_MAX_SPEED:
            self.calibration_start = None
            self.calibration_samples = []
            return

        if self.calibration_start is None:
            self.calibration_start = current_time
            self.calibration_samples = []

        self.calibration_samples.append(current_position)

        if current_time - self.calibration_start < NEUTRAL_CALIBRATION_TIME:
            return

        xs = [point[0] for point in self.calibration_samples]
        ys = [point[1] for point in self.calibration_samples]

        self.neutral_position = (
            sum(xs) / len(xs),
            sum(ys) / len(ys),
        )
        self.state = "READY"

        emit_event(
            self.event_outlet,
            "{}_NEUTRAL_CALIBRATED".format(self.side),
            lsl_timestamp,
        )

        print(
            "{} neutral = ({:.3f}, {:.3f})".format(
                self.side,
                self.neutral_position[0],
                self.neutral_position[1],
            )
        )

    def slowly_adapt_neutral(self, current_position, speed):
        if self.neutral_position is None or current_position is None or speed is None:
            return
        if speed > NEUTRAL_ADAPT_MAX_SPEED:
            return

        self.neutral_position = (
            NEUTRAL_ADAPT_ALPHA * current_position[0]
            + (1.0 - NEUTRAL_ADAPT_ALPHA) * self.neutral_position[0],
            NEUTRAL_ADAPT_ALPHA * current_position[1]
            + (1.0 - NEUTRAL_ADAPT_ALPHA) * self.neutral_position[1],
        )

    def update(self, current_position, speed, current_time, lsl_timestamp):
        """Advance the trial state machine; return distance from neutral"""
        if self.state == "UNCALIBRATED":
            self.update_calibration(
                current_position,
                speed,
                current_time,
                lsl_timestamp,
            )
            return None

        if self.neutral_position is None or current_position is None:
            return None

        distance_from_neutral = vector_distance(
            self.neutral_position,
            current_position,
        )

        if distance_from_neutral is None:
            return None

        if self.state == "READY":
            self.slowly_adapt_neutral(current_position, speed)

            if distance_from_neutral >= TRIAL_DEPARTURE_DISTANCE:
                self.trial_number += 1
                self.trial_start_time = current_time
                self.peak_distance = distance_from_neutral
                self.peak_emitted = False
                self.return_enter_time = None
                self.state = "ACTIVE"

                emit_event(
                    self.event_outlet,
                    "{}_TRIAL_MOVEMENT_ONSET".format(self.side),
                    lsl_timestamp,
                )

                print(
                    "{} trial {} started".format(
                        self.side,
                        self.trial_number,
                    )
                )

        elif self.state == "ACTIVE":
            if distance_from_neutral > self.peak_distance:
                self.peak_distance = distance_from_neutral

            if (
                not self.peak_emitted
                and self.peak_distance >= MIN_TRIAL_PEAK_DISTANCE
            ):
                self.peak_emitted = True
                emit_event(
                    self.event_outlet,
                    "{}_TRIAL_PEAK_REACHED".format(self.side),
                    lsl_timestamp,
                )

            if (
                distance_from_neutral <= TRIAL_RETURN_DISTANCE
                and self.peak_distance >= MIN_TRIAL_PEAK_DISTANCE
            ):
                self.return_enter_time = current_time
                self.state = "RETURNING"
                emit_event(
                    self.event_outlet,
                    "{}_TRIAL_RETURNED".format(self.side),
                    lsl_timestamp,
                )

        elif self.state == "RETURNING":
            if distance_from_neutral > TRIAL_RETURN_DISTANCE:
                self.state = "ACTIVE"
                self.return_enter_time = None
            else:
                stable_time = current_time - self.return_enter_time
                trial_duration = current_time - self.trial_start_time

                if (
                    stable_time >= TRIAL_RETURN_STABLE_TIME
                    and trial_duration >= MIN_TRIAL_DURATION
                ):
                    emit_event(
                        self.event_outlet,
                        "{}_TRIAL_COMPLETE".format(self.side),
                        lsl_timestamp,
                    )

                    print(
                        "{} trial {} complete | duration {:.2f}s | "
                        "peak {:.3f} body".format(
                            self.side,
                            self.trial_number,
                            trial_duration,
                            self.peak_distance,
                        )
                    )

                    self.state = "REARMING"
                    self.rearm_enter_time = current_time

        elif self.state == "REARMING":
            if distance_from_neutral > TRIAL_RETURN_DISTANCE:
                self.rearm_enter_time = current_time
            else:
                stable_time = current_time - self.rearm_enter_time

                if stable_time >= REARM_STABLE_TIME:
                    self.state = "READY"
                    self.peak_distance = 0.0
                    self.peak_emitted = False
                    self.trial_start_time = None
                    self.return_enter_time = None

                    emit_event(
                        self.event_outlet,
                        "{}_TRIAL_READY".format(self.side),
                        lsl_timestamp,
                    )

        return distance_from_neutral

    def is_active(self):
        return self.state in ("ACTIVE", "RETURNING")

    def is_ready(self):
        return self.state == "READY"


# ============================================================================
# EDGE NODE
# ============================================================================


class PoseEstimationNode(object):
    """Jetson PoseNet -> kinematics -> movement/trial events -> LSL"""

    JOINT_NAMES = (
        "L_SHOULDER",
        "L_ELBOW",
        "L_WRIST",
        "R_SHOULDER",
        "R_ELBOW",
        "R_WRIST",
    )

    def __init__(self, network, camera_uri, threshold, print_rate_hz, headless):
        self.network_name = network
        self.camera_uri = camera_uri
        self.threshold = threshold
        self.print_rate_hz = print_rate_hz
        self.headless = headless

        self.pose_outlet = create_pose_outlet(network, camera_uri)
        self.event_outlet = create_event_outlet()

        self.net = jetson_inference.poseNet(
            network,
            threshold=threshold,
        )

        self.camera = jetson_utils.videoSource(camera_uri)
        self.display = None

        if not headless:
            self.display = jetson_utils.videoOutput("display://0")

        self.keypoint_ids = {
            "L_SHOULDER": self.net.FindKeypointID("left_shoulder"),
            "L_ELBOW": self.net.FindKeypointID("left_elbow"),
            "L_WRIST": self.net.FindKeypointID("left_wrist"),
            "R_SHOULDER": self.net.FindKeypointID("right_shoulder"),
            "R_ELBOW": self.net.FindKeypointID("right_elbow"),
            "R_WRIST": self.net.FindKeypointID("right_wrist"),
        }

        self.smoothed = dict((name, None) for name in self.JOINT_NAMES)

        self.previous_time = None
        self.previous_wrist = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.velocity_smooth = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.motion_history = {
            "LEFT": deque(),
            "RIGHT": deque(),
        }

        self.movement = {
            "LEFT": MovementDetector("LEFT", self.event_outlet),
            "RIGHT": MovementDetector("RIGHT", self.event_outlet),
        }

        self.trial = {
            "LEFT": ArmTrialState("LEFT", self.event_outlet),
            "RIGHT": ArmTrialState("RIGHT", self.event_outlet),
        }

        self.last_print = 0.0
        self.last_person_seen = None
        self.tracking_reset_done = False

    def reset_temporal_tracking(self):
        """Clear stale kinematic/trial state, after long tracking loss"""
        self.smoothed = dict((name, None) for name in self.JOINT_NAMES)
        self.previous_time = None
        self.previous_wrist["LEFT"] = None
        self.previous_wrist["RIGHT"] = None
        self.velocity_smooth["LEFT"] = None
        self.velocity_smooth["RIGHT"] = None
        self.motion_history["LEFT"].clear()
        self.motion_history["RIGHT"].clear()
        self.movement["LEFT"].reset()
        self.movement["RIGHT"].reset()
        self.trial["LEFT"].reset_calibration()
        self.trial["RIGHT"].reset_calibration()

    def _running(self):
        if not self.camera.IsStreaming():
            return False
        if self.display is not None and not self.display.IsStreaming():
            return False
        return True

    def _raw_joint_points(self, pose):
        return dict(
            (
                name,
                get_keypoint(pose, self.keypoint_ids[name]),
            )
            for name in self.JOINT_NAMES
        )

    def _smooth_joints(self, raw):
        for name in self.JOINT_NAMES:
            self.smoothed[name] = smooth_point(
                self.smoothed[name],
                raw[name],
                POSITION_ALPHA,
            )

    def _append_motion_history(self, side, frame_time, relative_wrist):
        if relative_wrist is not None:
            self.motion_history[side].append((frame_time, relative_wrist))

        while (
            self.motion_history[side]
            and frame_time - self.motion_history[side][0][0] > MOTION_WINDOW + 0.25
        ):
            self.motion_history[side].popleft()

    def _build_pose_sample(
        self,
        left_angle,
        right_angle,
        left_relative,
        right_relative,
        left_velocity,
        right_velocity,
        left_window,
        right_window,
        shoulder_width,
        left_neutral_distance,
        right_neutral_distance,
    ):
        return [
            point_x(self.smoothed["L_SHOULDER"]),
            point_y(self.smoothed["L_SHOULDER"]),
            point_x(self.smoothed["L_ELBOW"]),
            point_y(self.smoothed["L_ELBOW"]),
            point_x(self.smoothed["L_WRIST"]),
            point_y(self.smoothed["L_WRIST"]),
            point_x(self.smoothed["R_SHOULDER"]),
            point_y(self.smoothed["R_SHOULDER"]),
            point_x(self.smoothed["R_ELBOW"]),
            point_y(self.smoothed["R_ELBOW"]),
            point_x(self.smoothed["R_WRIST"]),
            point_y(self.smoothed["R_WRIST"]),
            safe_value(left_angle),
            safe_value(right_angle),
            safe_value(left_relative[0]) if left_relative is not None else float("nan"),
            safe_value(left_relative[1]) if left_relative is not None else float("nan"),
            safe_value(right_relative[0]) if right_relative is not None else float("nan"),
            safe_value(right_relative[1]) if right_relative is not None else float("nan"),
            safe_value(left_velocity),
            safe_value(right_velocity),
            safe_value(left_window),
            safe_value(right_window),
            safe_value(shoulder_width),
            float(self.net.GetNetworkFPS()),
            safe_value(left_neutral_distance),
            safe_value(right_neutral_distance),
            safe_value(self.trial["LEFT"].peak_distance),
            safe_value(self.trial["RIGHT"].peak_distance),
            float(self.trial["LEFT"].is_active()),
            float(self.trial["RIGHT"].is_active()),
            float(self.trial["LEFT"].is_ready()),
            float(self.trial["RIGHT"].is_ready()),
        ]

    def _publish_no_person_sample(self, lsl_timestamp):
        sample = [float("nan")] * len(POSE_CHANNELS)
        sample[POSE_CHANNEL_INDEX["POSENET_FPS"]] = float(self.net.GetNetworkFPS())
        self.pose_outlet.push_sample(sample, timestamp=lsl_timestamp)

    def _print_status(
        self,
        frame_time,
        shoulder_width,
        left_angle,
        right_angle,
        left_velocity,
        right_velocity,
        left_neutral_distance,
        right_neutral_distance,
    ):
        if frame_time - self.last_print < (1.0 / self.print_rate_hz):
            return

        print()
        print("========================================")
        print("FPS: {:.1f}".format(self.net.GetNetworkFPS()))
        print("Shoulder width: {} px".format(fmt(shoulder_width)))
        print()

        print("LEFT")
        print("  state:         {}".format(self.trial["LEFT"].state))
        print("  trial #:       {}".format(self.trial["LEFT"].trial_number))
        print("  neutral dist:  {}".format(fmt(left_neutral_distance)))
        print("  peak dist:     {}".format(fmt(self.trial["LEFT"].peak_distance)))
        print("  elbow angle:   {} deg".format(fmt(left_angle)))
        print("  smooth speed:  {} body/s".format(fmt(left_velocity)))
        print("  raw moving:    {}".format(self.movement["LEFT"].moving))
        print()

        print("RIGHT")
        print("  state:         {}".format(self.trial["RIGHT"].state))
        print("  trial #:       {}".format(self.trial["RIGHT"].trial_number))
        print("  neutral dist:  {}".format(fmt(right_neutral_distance)))
        print("  peak dist:     {}".format(fmt(self.trial["RIGHT"].peak_distance)))
        print("  elbow angle:   {} deg".format(fmt(right_angle)))
        print("  smooth speed:  {} body/s".format(fmt(right_velocity)))
        print("  raw moving:    {}".format(self.movement["RIGHT"].moving))

        self.last_print = frame_time

    def process_pose(self, pose, frame_time, lsl_timestamp):
        self.last_person_seen = frame_time
        self.tracking_reset_done = False

        raw = self._raw_joint_points(pose)
        self._smooth_joints(raw)

        shoulder_width = point_distance(
            self.smoothed["L_SHOULDER"],
            self.smoothed["R_SHOULDER"],
        )

        left_relative = relative_point(
            self.smoothed["L_WRIST"],
            self.smoothed["L_SHOULDER"],
            shoulder_width,
        )
        right_relative = relative_point(
            self.smoothed["R_WRIST"],
            self.smoothed["R_SHOULDER"],
            shoulder_width,
        )

        left_angle = angle_at_joint(
            self.smoothed["L_SHOULDER"],
            self.smoothed["L_ELBOW"],
            self.smoothed["L_WRIST"],
        )
        right_angle = angle_at_joint(
            self.smoothed["R_SHOULDER"],
            self.smoothed["R_ELBOW"],
            self.smoothed["R_WRIST"],
        )

        dt = None if self.previous_time is None else frame_time - self.previous_time

        left_velocity_raw = normalized_velocity(
            self.previous_wrist["LEFT"],
            self.smoothed["L_WRIST"],
            dt,
            shoulder_width,
        )
        right_velocity_raw = normalized_velocity(
            self.previous_wrist["RIGHT"],
            self.smoothed["R_WRIST"],
            dt,
            shoulder_width,
        )

        self.velocity_smooth["LEFT"] = smooth_value(
            self.velocity_smooth["LEFT"],
            left_velocity_raw,
            VELOCITY_ALPHA,
        )
        self.velocity_smooth["RIGHT"] = smooth_value(
            self.velocity_smooth["RIGHT"],
            right_velocity_raw,
            VELOCITY_ALPHA,
        )

        self._append_motion_history("LEFT", frame_time, left_relative)
        self._append_motion_history("RIGHT", frame_time, right_relative)

        left_window = get_window_displacement(
            self.motion_history["LEFT"],
            frame_time,
        )
        right_window = get_window_displacement(
            self.motion_history["RIGHT"],
            frame_time,
        )

        self.movement["LEFT"].update(
            self.velocity_smooth["LEFT"],
            left_window,
            frame_time,
            lsl_timestamp,
        )
        self.movement["RIGHT"].update(
            self.velocity_smooth["RIGHT"],
            right_window,
            frame_time,
            lsl_timestamp,
        )

        left_neutral_distance = self.trial["LEFT"].update(
            left_relative,
            self.velocity_smooth["LEFT"],
            frame_time,
            lsl_timestamp,
        )
        right_neutral_distance = self.trial["RIGHT"].update(
            right_relative,
            self.velocity_smooth["RIGHT"],
            frame_time,
            lsl_timestamp,
        )

        pose_sample = self._build_pose_sample(
            left_angle,
            right_angle,
            left_relative,
            right_relative,
            self.velocity_smooth["LEFT"],
            self.velocity_smooth["RIGHT"],
            left_window,
            right_window,
            shoulder_width,
            left_neutral_distance,
            right_neutral_distance,
        )

        self.pose_outlet.push_sample(pose_sample, timestamp=lsl_timestamp)

        self._print_status(
            frame_time,
            shoulder_width,
            left_angle,
            right_angle,
            self.velocity_smooth["LEFT"],
            self.velocity_smooth["RIGHT"],
            left_neutral_distance,
            right_neutral_distance,
        )

        if self.smoothed["L_WRIST"] is not None:
            self.previous_wrist["LEFT"] = self.smoothed["L_WRIST"]
        if self.smoothed["R_WRIST"] is not None:
            self.previous_wrist["RIGHT"] = self.smoothed["R_WRIST"]

        self.previous_time = frame_time

    def process_tracking_loss(self, frame_time, lsl_timestamp):
        self._publish_no_person_sample(lsl_timestamp)

        if self.last_person_seen is None:
            self.last_person_seen = frame_time
            return

        if (
            not self.tracking_reset_done
            and frame_time - self.last_person_seen >= TRACKING_RESET_TIME
        ):
            self.reset_temporal_tracking()
            self.tracking_reset_done = True
            emit_event(self.event_outlet, "TRACKING_RESET", lsl_timestamp)

    def run(self):
        print()
        print("========================================")
        print(" ATS JETSON POSE ESTIMATION NODE")
        print("========================================")
        print()
        print("Model:      {}".format(self.network_name))
        print("Camera:     {}".format(self.camera_uri))
        print("Threshold:  {:.3f}".format(self.threshold))
        print("Pose LSL:   {} ({} channels)".format(POSE_STREAM_NAME, len(POSE_CHANNELS)))
        print("Event LSL:  {}".format(EVENT_STREAM_NAME))
        print("Display:    {}".format("off" if self.headless else "on"))
        print()
        print("Stand still in neutral position.")
        print(
            "Neutral calibration takes {:.1f} seconds.".format(
                NEUTRAL_CALIBRATION_TIME
            )
        )
        print()

        while self._running():
            image = self.camera.Capture()

            if image is None:
                continue

            # Separate clocks deliberately have separate purposes:
            # Monotonic for elapsed-time calculations. 
            # LSL for recorded timing.
            frame_time = time.monotonic()
            frame_lsl_time = local_clock()

            poses = self.net.Process(
                image,
                overlay="links,keypoints",
            )

            if len(poses) > 0:
                # Current version tracks first PoseNet detection.
                self.process_pose(
                    poses[0],
                    frame_time,
                    frame_lsl_time,
                )
            else:
                self.process_tracking_loss(
                    frame_time,
                    frame_lsl_time,
                )

            if self.display is not None:
                self.display.Render(image)
                self.display.SetStatus(
                    "ATS Pose Node | {:.1f} FPS".format(
                        self.net.GetNetworkFPS()
                    )
                )


# ============================================================================
# COMMAND LINE
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ATS Jetson PoseNet edge node; publish body-pose and vision-event streams over LSL."
        )
    )

    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK,
        help="PoseNet model (default: {}).".format(DEFAULT_NETWORK),
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="jetson_utils videoSource URI (default: {}).".format(DEFAULT_CAMERA),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="PoseNet confidence threshold (default: {}).".format(DEFAULT_THRESHOLD),
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=DEFAULT_PRINT_RATE_HZ,
        help="Terminal status updates per second (default: {}).".format(DEFAULT_PRINT_RATE_HZ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable local display window while continuing LSL publication.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.threshold <= 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be greater than 0 and at most 1.")
    if args.print_rate <= 0.0:
        raise ValueError("--print-rate must be greater than zero.")

    node = PoseEstimationNode(
        network=args.network,
        camera_uri=args.camera,
        threshold=args.threshold,
        print_rate_hz=args.print_rate,
        headless=args.headless,
    )

    try:
        node.run()
    except KeyboardInterrupt:
        print()
        print("ATS Jetson pose estimation node stopped.")


if __name__ == "__main__":
    main()
