# Multimodal Neural-Motor Sensing

A multimodal experimental platform for synchronising neural signals, behavioural events and vision-derived human motion within a common recording and analysis pipeline.

The system was developed to explore how temporally aligned biological and physical measurements can be used to study human motor behaviour and, ultimately, support multimodal state-estimation systems.

> **Status:** Active experimental prototype. Current datasets are exploratory and are not intended to support neuroscientific conclusions.

<p align="center">
  <img src="media/multimodal_timeline.jpg" width="100%" alt="Synchronised multimodal EEG and movement recording">
</p>

*Example synchronised recording showing EEG alongside vision-derived upper-limb kinematics and behavioural event markers.*

---

## System Overview

The platform combines multiple independently generated data streams:

- 5-channel EEG acquisition
- timestamped experimental event markers
- vision-derived upper-body kinematics
- behavioural movement validation
- synchronised multimodal recording
- event-aligned EEG analysis

Streams are synchronised using **Lab Streaming Layer (LSL)** and recorded into **XDF**, allowing neural, behavioural and motion-derived measurements to be referenced against a common time base.

Recorded EEG data can subsequently be imported into **MNE-Python** for filtering, epoch generation, spectral analysis and time-frequency analysis.

### Current Data Flow

```text
                  ┌────────────────────┐
                  │   EEG Headset      │
                  │  5-channel EEG     │
                  └─────────┬──────────┘
                            │
                            ▼
                       EEG LSL Stream
                            │
                            │
┌──────────────────┐        │        ┌─────────────────────┐
│ Experiment / Cue │        │        │ Camera / Vision     │
│ Event Generator  │        │        │ Motion Estimation   │
└────────┬─────────┘        │        └──────────┬──────────┘
         │                  │                   │
         ▼                  ▼                   ▼
   Marker Stream ───────► LSL / XDF ◄──── Motion Stream
                            │
                            ▼
                  ┌────────────────────┐
                  │ Trial Validation   │
                  │ & Event Alignment  │
                  └─────────┬──────────┘
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
             Cue-aligned          Movement-aligned
                epochs                epochs
                  │                    │
                  └─────────┬──────────┘
                            ▼
                       MNE Analysis
                            │
             ┌──────────────┼───────────────┐
             ▼              ▼               ▼
           EEG          PSD Analysis    Time-Frequency
         Averaging                        Analysis
```

---

## Current Capabilities

The current prototype supports:

- live EEG acquisition and streaming
- experiment cue and event generation
- synchronised LSL recording
- XDF multimodal data storage
- camera-derived upper-limb motion measurements
- automatic movement-onset detection
- behavioural validation of commanded trials
- reaction-time calculation
- cue-aligned EEG epoch generation
- movement-onset-aligned EEG epoch generation
- EEG filtering and mains-frequency rejection
- power spectral density analysis
- time-frequency / ERD-ERS visualisation

---

## Example Experiment

A motor-response experiment was used to test the end-to-end pipeline.

The system presented LEFT or RIGHT movement cues and recorded the corresponding EEG and observed movement. Trials were subsequently validated by checking whether the detected movement matched the commanded action.

A recent exploratory recording contained:

```text
Commanded trials:       6
Behaviourally valid:    4
LEFT trials:            3
RIGHT trials:           1

Mean reaction time:     0.6885 s
Minimum:                0.1543 s
Maximum:                1.0136 s
```

The current dataset is intentionally small and is used to validate the acquisition and analysis architecture rather than draw neuroscientific conclusions.

---

## EEG Processing

Current exploratory processing includes:

```text
Band-pass:    1–40 Hz
Mains notch:  50 Hz
```

Validated trials can be transformed into both cue-aligned and movement-onset-aligned MNE Epochs.

This allows neural activity to be compared against both the intended experimental event and the independently observed physical response.

---

## Why Multimodal?

EEG alone provides an incomplete description of behaviour.

By recording neural activity alongside observed physical movement and explicit experiment events, the system can distinguish between:

```text
command issued
      ↓
neural activity
      ↓
movement initiation
      ↓
observed physical motion
```

This provides a foundation for future work in multimodal state estimation, sensor fusion and neural-motor modelling.

---

## Technologies

### Languages

- Python

### Acquisition & Synchronisation

- Lab Streaming Layer (LSL)
- XDF
- LabRecorder

### Signal Processing

- MNE-Python
- NumPy
- SciPy

### Visualisation

- Matplotlib

### Sensing

- EEG
- computer vision / pose-derived kinematics
- timestamped behavioural events

---

## Current Limitations

This is an experimental engineering platform rather than a validated neuroscience study.

Current limitations include:

- small experimental datasets
- limited EEG channel count
- movement measurements derived from camera-based pose estimation rather than laboratory motion capture
- software-level stream synchronisation rather than dedicated hardware timestamping
- behavioural validation does not currently constitute EEG artefact rejection

These limitations are being treated as engineering constraints for future iterations rather than hidden from the analysis.

---

## Planned Development

Future work includes:

- additional inertial sensing
- embedded-system telemetry
- improved stream timing characterisation
- quantitative synchronisation-error measurement
- larger balanced experimental datasets
- improved artefact rejection
- additional movement and behavioural classes
- multimodal feature extraction
- learned sensor-fusion models
- real-time multimodal state estimation

---

## Project Context

This project developed from earlier work on a real-time EEG-controlled mobile robot and is part of a broader investigation into systems that connect biological sensing, perception, embedded computation and physical behaviour.

The immediate focus is the engineering infrastructure required to acquire, synchronise, validate and analyse heterogeneous sensor streams reliably.
