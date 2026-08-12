# NVIDIA Jetson Edge Perception Node

The multimodal sensing platform includes an NVIDIA Jetson Nano as a
networked edge-compute node for camera-based human motion estimation.

Its purpose is to convert a live camera stream into synchronised
kinematic measurements and movement events that can be recorded
alongside EEG and experiment markers using Lab Streaming Layer (LSL).

The implementation is provided in:

```text
edge/pose_estimation_node.py
```

---

## Overview

The Jetson node performs the following processing chain:

```text
CSI Camera
    │
    ▼
NVIDIA PoseNet
    │
    ▼
Upper-body keypoints
    │
    ├── shoulder
    ├── elbow
    └── wrist
    │
    ▼
Temporal smoothing
    │
    ▼
Body-normalised kinematics
    │
    ├── wrist position
    ├── wrist speed
    ├── rolling displacement
    └── elbow angle
    │
    ├───────────────────────┐
    │                       │
    ▼                       ▼
Movement detector       Trial state machine
    │                       │
    └───────────┬───────────┘
                │
                ▼
        ATS_VISION_EVENTS

                +

          ATS_BODY_POSE
                │
                ▼
        Lab Streaming Layer
                │
                ▼
       Networked XDF recorder
```

This separates computationally expensive vision processing from the
EEG acquisition and experiment-control processes while allowing all
outputs to participate in the same multimodal recording architecture.

---

## Edge Hardware

The current implementation targets:

```text
Platform:       NVIDIA Jetson Nano
Camera input:   NVIDIA jetson_utils videoSource
Pose model:     resnet18-body
Inference API:  jetson-inference PoseNet
LSL interface:  pylsl
```

The default camera URI is:

```text
csi://0
```

and the default PoseNet confidence threshold is:

```text
0.15
```

The node was intentionally kept compatible with the Python environment
available on the deployed Jetson Nano.

NVIDIA-specific dependencies such as `jetson-inference` and
`jetson-utils` are therefore treated as platform dependencies rather
than normal desktop Python packages.

---

## Running the Node

Default configuration:

```bash
python edge/pose_estimation_node.py
```

Specify a different camera:

```bash
python edge/pose_estimation_node.py --camera csi://1
```

Specify the PoseNet model and confidence threshold:

```bash
python edge/pose_estimation_node.py \
    --network resnet18-body \
    --threshold 0.15
```

The local visualisation can be disabled for headless operation:

```bash
python edge/pose_estimation_node.py --headless
```

LSL publication continues when the display is disabled.

---

# Published LSL Streams

The edge node publishes two independent streams.

## `ATS_BODY_POSE`

```text
Type:           Pose
Channel count:  32
Sample rate:    irregular
Data type:      float32
```

The irregular sample rate reflects the fact that samples are produced
as camera frames pass through the pose-estimation pipeline rather than
from a fixed-frequency physical sensor clock.

### Channel groups

| Group | Channels | Units |
|---|---:|---|
| Camera-space shoulder/elbow/wrist coordinates | 12 | pixels |
| Elbow angles | 2 | degrees |
| Body-relative wrist coordinates | 4 | shoulder widths |
| Wrist speed | 2 | shoulder widths / second |
| Rolling wrist displacement | 2 | shoulder widths |
| Body scale and inference performance | 2 | pixels / frames per second |
| Neutral-position distance | 2 | shoulder widths |
| Trial peak displacement | 2 | shoulder widths |
| Trial-active state | 2 | boolean |
| Trial-ready state | 2 | boolean |

The complete channel schema is embedded into the LSL stream metadata so
recording and analysis tools can recover channel labels and units from
the recording.

---

## `ATS_VISION_EVENTS`

```text
Type:           Markers
Channel count:  1
Sample rate:    irregular
Data type:      string
```

This stream contains discrete events derived from the vision pipeline.

### Raw movement events

```text
LEFT_MOVEMENT_START
LEFT_MOVEMENT_STOP

RIGHT_MOVEMENT_START
RIGHT_MOVEMENT_STOP
```

These events are used by the behavioural-validation pipeline to compare
commanded movements against independently observed physical motion.

### Trial-state events

The higher-level movement state machine can also emit:

```text
LEFT_NEUTRAL_CALIBRATED
RIGHT_NEUTRAL_CALIBRATED

LEFT_TRIAL_MOVEMENT_ONSET
RIGHT_TRIAL_MOVEMENT_ONSET

LEFT_TRIAL_PEAK_REACHED
RIGHT_TRIAL_PEAK_REACHED

LEFT_TRIAL_RETURNED
RIGHT_TRIAL_RETURNED

LEFT_TRIAL_COMPLETE
RIGHT_TRIAL_COMPLETE

LEFT_TRIAL_READY
RIGHT_TRIAL_READY
```

A further:

```text
TRACKING_RESET
```

event is emitted when a sustained loss of body tracking causes temporal
movement state to be cleared.

---

# Pose Processing

## Keypoint Extraction

PoseNet provides body keypoints for detected people.

The current node extracts:

```text
left shoulder
left elbow
left wrist

right shoulder
right elbow
right wrist
```

The current implementation tracks the first detected PoseNet person.

Persistent identity association for multi-person scenes is not currently
implemented.

---

## Temporal Smoothing

Raw image-space joint coordinates are filtered using an exponential
moving average.

Current position smoothing:

```text
alpha = 0.35
```

Derived wrist velocity is separately smoothed using:

```text
alpha = 0.30
```

This reduces frame-to-frame pose-estimation jitter before movement
features are calculated.

---

# Body-Normalised Kinematics

Image-space displacement depends on factors including subject distance
from the camera.

To reduce this dependency, the detected shoulder separation is used as
an approximate body-scale reference.

For each frame:

```text
shoulder width
      │
      ▼
wrist position relative to same-side shoulder
      │
      ▼
divide by shoulder width
      │
      ▼
body-relative wrist position
```

Motion can therefore be expressed in approximately:

```text
shoulder widths
```

rather than only pixels.

Wrist velocity is similarly expressed as:

```text
shoulder widths / second
```

A minimum valid shoulder width is enforced before body-normalised
measurements are accepted.

This is intended as a practical normalisation strategy for the
experimental vision pipeline rather than a calibrated metric
motion-capture measurement.

---

# Movement Detection

Movement detection combines two complementary measures.

## Instantaneous wrist speed

A movement can begin when smoothed wrist speed exceeds:

```text
0.38 shoulder widths / second
```

## Rolling displacement

Slow but sustained movement can otherwise be missed by a simple speed
threshold.

The node therefore also evaluates displacement over a rolling:

```text
0.50 second
```

window.

A movement may begin when rolling displacement exceeds:

```text
0.10 shoulder widths
```

---

## Hysteresis

Different thresholds are used for movement start and movement stop.

Current stop thresholds are:

```text
speed:               0.18 shoulder widths / second
rolling displacement: 0.035 shoulder widths
```

The motion must remain below the stop criteria for:

```text
0.30 seconds
```

before a movement-stop event is generated.

This hysteresis reduces rapid switching between moving and stationary
states near a single threshold.

---

# Higher-Level Trial Detection

In addition to raw motion detection, each arm maintains an independent
trial state machine.

```text
UNCALIBRATED
      │
      ▼
    READY
      │
      ▼
   ACTIVE
      │
      ▼
 RETURNING
      │
      ▼
 REARMING
      │
      └────────────► READY
```

## Neutral calibration

The user initially remains relatively still while the node estimates a
neutral wrist position.

Current calibration duration:

```text
2.0 seconds
```

Calibration restarts if excessive movement is detected during this
period.

The neutral estimate can subsequently adapt slowly while the arm is
stationary, allowing limited compensation for gradual postural drift.

---

## Trial departure

A trial begins when the wrist moves sufficiently far from its calibrated
neutral position.

The system subsequently tracks:

```text
movement onset
peak displacement
return toward neutral
stable return
trial completion
re-arm
```

The return movement therefore remains part of the same physical trial
rather than incorrectly becoming a second trial.

---

# Tracking Loss

Camera-derived motion systems must explicitly handle temporary loss of
the subject or individual keypoints.

When no person is detected, the pose stream continues with unavailable
measurements represented as `NaN`.

If tracking remains lost for longer than the configured reset period,
the node clears:

```text
smoothed joint history
previous wrist positions
velocity history
rolling motion history
movement state
neutral calibration
active trial state
```

and emits:

```text
TRACKING_RESET
```

This prevents a newly reacquired subject position from being compared
against stale pre-loss coordinates and producing a false velocity or
movement event.

---

# Timing Architecture

Timing is intentionally separated into two concepts.

## Internal elapsed time

Python's monotonic clock is used for calculations involving elapsed
duration, including:

```text
frame-to-frame velocity
movement stop delay
neutral calibration
trial duration
stable-return timing
re-arm timing
tracking-loss timeout
```

A monotonic clock is appropriate for these calculations because it is
not affected by wall-clock corrections.

---

## Recorded LSL time

Immediately after a camera frame is returned by the capture API, the
node obtains one LSL clock timestamp.

That timestamp is then associated with:

```text
the ATS_BODY_POSE sample derived from that frame

and

any ATS_VISION_EVENTS generated from that frame
```

This means a detected movement event and the pose sample responsible for
that event use the same software-side LSL timestamp.

```text
Camera.Capture()
      │
      ▼
LSL timestamp assigned
      │
      ├── PoseNet / derived kinematics
      │
      ├── movement detector
      │
      └── trial detector
              │
              ▼
      shared frame timestamp
```

This reduces avoidable software-side timing differences between
kinematic samples and their derived events.

---

## Important Timing Limitation

The timestamp assigned by this node is **not a camera sensor hardware
timestamp**.

It represents the software-side time immediately after the captured
frame becomes available to the application.

The total camera-to-timestamp delay can therefore include:

```text
sensor exposure
camera transport
capture buffering
driver / GStreamer processing
OS scheduling
```

The current system consequently provides software-level multimodal
synchronisation rather than fully characterised hardware-level
synchronisation.

Quantitative measurement of this residual timing uncertainty is planned
future work.

---

# Networked Operation

The Jetson operates as an independent LSL host.

A second computer on the same network can discover its streams using:

```bash
python tools/discover_lsl_streams.py --ats-only
```

The discovery utility reports the publishing host alongside each stream,
making it possible to distinguish streams generated by the workstation
from those generated by the Jetson edge node.

A typical distributed experiment can therefore have:

```text
WORKSTATION
├── ATS_EEG_RAW
├── ATS_MARKERS
└── ATS_EXPERIMENT

JETSON EDGE NODE
├── ATS_BODY_POSE
└── ATS_VISION_EVENTS
```

All selected streams can then be recorded into the same XDF session.

---

# Relationship to Behavioural Validation

The vision system provides an independent observation of physical
movement.

The experimental task provides the commanded action:

```text
LEFT
or
RIGHT
```

while the Jetson generates events such as:

```text
LEFT_MOVEMENT_START
RIGHT_MOVEMENT_START
```

The validation stage can therefore compare:

```text
commanded movement
       │
       ▼
vision-observed movement
       │
       ▼
reaction time / trial validity
```

This prevents the EEG analysis from assuming that a commanded movement
was actually performed correctly.

---

# Current Limitations

The edge perception node is an experimental engineering system rather
than a calibrated motion-capture platform.

Current limitations include:

- 2-D image-space pose estimation rather than metric 3-D reconstruction
- body-scale normalisation based on apparent shoulder width
- first-detected-person tracking rather than persistent subject identity
- software-side camera timestamps rather than sensor hardware timestamps
- model inference and camera buffering contribute uncharacterised latency
- movement thresholds are experimentally selected rather than
  population-validated
- camera tracking can be affected by occlusion, lighting, body
  orientation and pose-estimation confidence

These limitations are exposed explicitly so that subsequent iterations
can quantify or remove them.

---

# Planned Development

Potential extensions include:

- quantitative camera-to-LSL latency measurement
- persistent subject association
- hardware-assisted event synchronisation
- additional camera views
- inertial-sensor fusion
- embedded-system telemetry
- richer movement classes
- GPU inference benchmarking
- learned multimodal state estimation
- automatic node health and stream-quality monitoring

The current implementation establishes the distributed sensing
architecture required for those extensions.
