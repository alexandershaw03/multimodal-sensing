# Multimodal Neural-Motor Sensing

A platform for synchronising neural signals, behavioural events and vision-derived human motion in one recording and analysis pipeline. I have built it to study how temporally-aligned biological and physical measurements can describe and display human motor and neural behaviour.

> **Status:** Active experimental prototype. Current datasets are exploratory, not neuroscientific conclusions.

[![Synchronised multimodal EEG and movement recording](https://github.com/alexandershaw03/multimodal-sensing/raw/main/media/multimodal_timeline.jpg)](/alexandershaw03/multimodal-sensing/blob/main/media/multimodal_timeline.jpg)

*Example synchronised recording: EEG, alongside vision-derived upper-limb kinematics and behavioural event markers.*

---

## System Overview

Independently generated neural, behavioural and vision streams are combined using **Lab Streaming Layer (LSL)**. Selected streams recorded together as **XDF**, tto get validated and analysed offline with **MNE-Python**.

```
                              WORKSTATION

                    ┌────────────────────┐
                    │   Emotiv Insight   │
                    │    5-ch EEG        │
                    └─────────┬──────────┘
                              │ Cortex
                              ▼
                       ATS_EEG_RAW
                              │
                              │
┌────────────────────┐        │        ┌─────────────────────┐
│ Experiment / Cues  │        │        │ Manual annotation   │
│  ATS_EXPERIMENT    │        │        │    ATS_MARKERS      │
└──────────┬─────────┘        │        └──────────┬──────────┘
           │                  │                   │
           └──────────────────┼───────────────────┘
                              │
                              │ LSL network
                              │
              ┌───────────────┴────────────────┐
              │                                │
              │                        NVIDIA JETSON NANO
              │
              │                     ┌─────────────────────┐
              │                     │ CSI camera          │
              │                     │ PoseNet inference   │
              │                     │ kinematics          │
              │                     │ movement detection  │
              │                     └──────────┬──────────┘
              │                                │
              │                     ┌──────────┴──────────┐
              │                     ▼                     ▼
              │              ATS_BODY_POSE       ATS_VISION_EVENTS
              │                     │                     │
              └─────────────────────┴─────────────────────┘
                                    │
                                    ▼
                               LSL / XDF
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
              Trial validation             Multimodal plotting
                     │
                     ▼
              Validated trials
                     │
              ┌──────┴────────┐
              ▼               ▼
         Cue-aligned     Movement-aligned
            epochs           epochs
              │               │
              └──────┬────────┘
                     ▼
                  MNE analysis
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
      EEG averages   PSD     Time-frequency
```

The Jetson is an optional, networked edge node; workstation-side acquisition, experiment control and offline analysis, all work independently of it.

---

## What It Does

- live 5-ch EEG acquisition (Emotiv Insight, via Cortex API) - `ATS_EEG_RAW`
- manual timestamped markers (`ATS_MARKERS`) and randomised LEFT/RIGHT motor-response experiments (`ATS_EXPERIMENT`)
- Jetson-side pose estimation (PoseNet) - 32-channel body pose and movement events (`ATS_BODY_POSE`, `ATS_VISION_EVENTS`), including wrist coordinates/velocity, elbow-angle estimation and automatic neutral-pose calibration
- cross-machine LSL stream-discovery and XDF recording/inspection
- behavioural validation of commanded trials - against observed movement, with reaction-time calculation
- cue and movement-onset-aligned EEG epoching, filtering, PSD and time-frequency/ERD-ERS analysis

---

## LSL Streams

| Stream              | Host        | Type    | Contents                                           |
| ------------------- | ----------- | ------- | -------------------------------------------------- |
| `ATS_EEG_RAW`       | Workstation | EEG     | AF3, T7, Pz, T8, AF4 at nominal 128 Hz             |
| `ATS_MARKERS`       | Workstation | Markers | Manual behavioural / experimental annotations      |
| `ATS_EXPERIMENT`    | Workstation | Markers | Automatic motor-task cues, phases and trial events |
| `ATS_BODY_POSE`     | Jetson      | Pose    | 32-channel pose, kinematic and trial-state data    |
| `ATS_VISION_EVENTS` | Jetson      | Markers | Vision-derived movement and trial-state events     |

Irregular streams are timestamped per-sample/event, rather than assigned a fixed nominal rate.

---

## Quick Start

### 1. Clone

```
git clone https://github.com/alexandershaw03/multimodal-sensing.git
cd multimodal-sensing
```

### 2. Workstation environment (Python 3.10+)

**Windows PowerShell**
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Jetson edge node uses NVIDIA platform libraries and is documented separately in [`docs/jetson_edge_node.md`](https://github.com/alexandershaw03/multimodal-sensing/blob/main/docs/jetson_edge_node.md).

### 3. Cortex credentials

```
Copy-Item .env.example .env      # Windows
cp .env.example .env             # Linux / macOS
```

```
EMOTIV_CLIENT_ID=your_client_id
EMOTIV_CLIENT_SECRET=your_client_secret
```

Optionally set a local data directory: `ATS_DATA_ROOT=C:\path\to\your\data`


`.env` is gitignored ... **if forked, don't commit *your* Cortex credentials!** Additionally, EEG-acquisition needs local EMOTIV-Cortex services running (e.g. EMOTIV Launcher), connected to an authorised EMOTIV headset.  

*Note: This adds to the reasons why I am transitioning from an EMOTIV headset, so EEG data can be streamed offline and independantly from manufacturer software*

---

## Running It

**Live EEG:**
```
python acquisition/eeg_lsl_stream.py           # --debug for diagnostics
```
Publishes `ATS_EEG_RAW` (AF3, T7, Pz, T8, AF4 @ 128 Hz nominal).

**Manual markers:**
```
python acquisition/marker_stream.py
```
Publishes `ATS_MARKERS`. Commands: `REST`, `LEFT_ARM`, `RIGHT_ARM`, `FORWARD_REACH`, `BLINK`, `STOP`.

**Automatic motor experiment:**
```
python experiments/motor_task.py
```
Publishes `ATS_EXPERIMENT`, writes a CSV backup log (under `ATS_DATA_ROOT` unless `--log-dir` is given).

**Jetson edge perception:**
```
python edge/pose_estimation_node.py            # --headless for no display
```
Runs PoseNet inference, derives kinematics, publishes `ATS_BODY_POSE` / `ATS_VISION_EVENTS`. Full setup, timing and schema in [`docs/jetson_edge_node.md`](https://github.com/alexandershaw03/multimodal-sensing/blob/main/docs/jetson_edge_node.md).

**Discover live streams (any machine on the network):**
```
python tools/discover_lsl_streams.py --ats-only
```

---

## Recording

Record the live streams together with **LabRecorder** or another XDF-compatible LSL recorder. A typical motor experiment: `ATS_EEG_RAW`, `ATS_EXPERIMENT`, `ATS_BODY_POSE`, `ATS_VISION_EVENTS` (plus `ATS_MARKERS` for manual annotation).

Raw participant recordings are intentionally excluded from this public repo.

---

## Inspecting and Analysing XDF

```
python tools/inspect_xdf.py recording.xdf
```
Stream names, dimensions, duration, nominal vs. timestamp-derived rates.

```
python analysis/plot_multimodal_xdf.py recording.xdf
python analysis/plot_multimodal_xdf.py recording.xdf --start 20 --end 50 --show
```
Supports current `ATS_EXPERIMENT` events and legacy `ATS_MARKERS`, resolving pose channels from the recorded schema.

```
python validation/validate_motor_trials.py recording.xdf
python validation/validate_motor_trials.py recording.xdf --require-down-valid
```
Compares commanded LEFT/RIGHT trials against detected vision events, writes `recording_validated_trials.csv`.

```
python analysis/analyse_validated_motor_eeg.py recording.xdf --show
```
Auto-discovers the validation CSV if named by default. Outputs: reaction-time plot, cue- and movement-aligned EEG averages and PSD, time-frequency maps, MNE Epochs, summary.

```
python analysis/xdf_to_mne.py recording.xdf
```
Converts `ATS_EEG_RAW` + `ATS_MARKERS` recordings to MNE `.fif`, with `--show` for the raw-data viewer.

---

## Repository Structure

```
multimodal-sensing/
│
├── acquisition/
│   ├── eeg_lsl_stream.py
│   └── marker_stream.py
│
├── experiments/
│   └── motor_task.py
│
├── edge/
│   └── pose_estimation_node.py
│
├── validation/
│   └── validate_motor_trials.py
│
├── analysis/
│   ├── xdf_to_mne.py
│   ├── plot_multimodal_xdf.py
│   └── analyse_validated_motor_eeg.py
│
├── tools/
│   ├── discover_lsl_streams.py
│   ├── inspect_xdf.py
│   └── read_eeg_stream.py
│
├── docs/
│   ├── synchronisation.md
│   └── jetson_edge_node.md
│
├── media/
├── results/
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Representative Results

Included as evidence the pipeline runs end-to-end — acquisition through synchronisation, validation, and analysis. The recording below is exploratory, not a neuroscience result.

**Behavioural validation**

[![Validated motor trial reaction times](https://github.com/alexandershaw03/multimodal-sensing/raw/main/results/01_reaction_times.png)](/alexandershaw03/multimodal-sensing/blob/main/results/01_reaction_times.png)

4 of 6 commanded trials were behaviourally validated. Movement onset was estimated independently from the motion stream, giving cue-to-movement reaction time.

**Cue-aligned EEG**

[![Cue-aligned EEG](https://github.com/alexandershaw03/multimodal-sensing/raw/main/results/02_cue_aligned_eeg.png)](/alexandershaw03/multimodal-sensing/blob/main/results/02_cue_aligned_eeg.png)

**Movement-onset-aligned EEG**

[![Movement-onset-aligned EEG](https://github.com/alexandershaw03/multimodal-sensing/raw/main/results/03_movement_aligned_eeg.png)](/alexandershaw03/multimodal-sensing/blob/main/results/03_movement_aligned_eeg.png)

Same validated trials, aligned to detected physical movement onset rather than the commanded cue.

**Spectral analysis**

[![Movement-aligned EEG power spectral density](https://github.com/alexandershaw03/multimodal-sensing/raw/main/results/04_movement_psd.png)](/alexandershaw03/multimodal-sensing/blob/main/results/04_movement_psd.png)

More time-frequency analyses in [`results/`](https://github.com/alexandershaw03/multimodal-sensing/blob/main/results).

### Example experiment

LEFT/RIGHT movement cues were presented and the corresponding EEG and observed movement recorded, then validated by checking detected movement against the commanded action:

```
Commanded trials:       6
Behaviourally valid:    4
LEFT trials:            3
RIGHT trials:           1
Mean reaction time:     0.6885 s
Minimum:                0.1543 s
Maximum:                1.0136 s
```

Deliberately a small dataset. Performed in <10mins, I have included it here to validate the pipeline, not to draw conclusions from.

---

## EEG Processing

Band-pass 1–40 Hz, 50 Hz mains notch. Validated trials convert to both cue-aligned and movement-onset-aligned MNE Epochs, so neural activity can be compared against the intended event *and* the observed physical response. Note: behavioural validation confirms the movement happened — it isn't EEG artefact rejection.

---

## Synchronisation

Currently software-level LSL synchronisation — EEG, experiment, marker and vision processes share a common LSL/XDF timing framework, but there's no shared hardware trigger across every sensor yet. For the Jetson vision node specifically, the LSL timestamp is applied software-side to a processed frame, not read from camera hardware. That leaves some timing uncertainty from transport latency, camera buffering, vision processing, OS scheduling and network timing — quantifying it end-to-end is on the to-do list (see below). Details in [`docs/synchronisation.md`](https://github.com/alexandershaw03/multimodal-sensing/blob/main/docs/synchronisation.md).

---

## Technologies

**Acquisition/sync:** LSL, XDF, LabRecorder, Emotiv Cortex API
**Edge:** NVIDIA Jetson Nano, jetson-inference, PoseNet, TensorRT
**Processing:** MNE-Python, NumPy, SciPy, pandas, Matplotlib

---

## Limitations & Next Steps

Biggest open items right now: the dataset is small; pose/s comes from 2D-vision, rather than calibrated motion capture; synchronisation is software-level (no shared hardware trigger...yet); person-tracking is first-detected-only, rather than persistent identity — fine for at-home one-person trials, not for multi-person scenes.

Next up: quantifying end-to-end sync error, benchmarking Jetson latency/throughput, moving toward hardware-assisted event sync.  
Longer-term: IMU fusion; larger, balanced, datasets; learned multimodal state-estimation, rather than hand-picked thresholds.  
I will do my best to keep these tracked through GitHub Issues as it goes.

---

This grew out of the [`eeg-robot`](https://github.com/alexandershaw03/eeg-robot) project. As my undergrad project, and once I had EEG reliably driving a robot, the natural next question was what else could be measured alongside it, and how far can I take it. I hope this repository can be evidence of that, potentially also alluding to some of my more ambitious ideas too.

---

## License

MIT — see [`LICENSE`](https://github.com/alexandershaw03/multimodal-sensing/blob/main/LICENSE).
