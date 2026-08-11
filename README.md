# Multimodal Neural-Motor Sensing

A multimodal experimental platform for synchronising neural signals, behavioural events and vision-derived human motion within a common recording and analysis pipeline.

The system was developed to explore how temporally aligned biological and physical measurements can be used to study human motor behaviour and, ultimately, support multimodal state-estimation systems.

> **Status:** Active experimental prototype. (Current datasets are exploratory and are not intended to support neuroscientific conclusions.)

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

### Current data flow

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
