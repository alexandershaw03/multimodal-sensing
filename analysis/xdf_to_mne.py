import pyxdf
import numpy as np
import mne
from pathlib import Path


# =========================================================
# FILE
# =========================================================

# Change "xxx"'s to actual file path
XDF_FILE = Path(
    r" xxx "
)

CHANNELS = [
    'AF3',
    'T7',
    'Pz',
    'T8',
    'AF4'
]

SFREQ = 128.0


print("\n========================================")
print(" ATS XDF -> MNE")
print("========================================\n")

print(f"Loading:\n{XDF_FILE}\n")

streams, header = pyxdf.load_xdf(str(XDF_FILE))


# =========================================================
# FIND STREAMS
# =========================================================

eeg_stream = None
marker_stream = None

for stream in streams:

    name = stream['info']['name'][0]

    if name == 'ATS_EEG_RAW':
        eeg_stream = stream

    elif name == 'ATS_MARKERS':
        marker_stream = stream


if eeg_stream is None:
    raise RuntimeError("ATS_EEG_RAW not found.")

if marker_stream is None:
    raise RuntimeError("ATS_MARKERS not found.")


# =========================================================
# EEG
# =========================================================

eeg = np.asarray(
    eeg_stream['time_series'],
    dtype=float
)

eeg_timestamps = np.asarray(
    eeg_stream['time_stamps'],
    dtype=float
)

print(f"EEG samples:  {eeg.shape[0]}")
print(f"EEG channels: {eeg.shape[1]}")


# Cortex EEG values are in microvolts but MNE expects EEG in VOLTS.
#
# µV -> V
eeg_volts = eeg / 1_000_000.0


# MNE expects:
# channels x samples
#
# rather than:
# samples x channels

eeg_volts = eeg_volts.T


# =========================================================
# CREATE MNE INFO
# =========================================================

info = mne.create_info(
    ch_names=CHANNELS,
    sfreq=SFREQ,
    ch_types='eeg'
)


# =========================================================
# CREATE RAW OBJECT
# =========================================================

raw = mne.io.RawArray(
    eeg_volts,
    info
)


# =========================================================
# ADD ELECTRODE POSITIONS
# =========================================================

montage = mne.channels.make_standard_montage(
    'standard_1020'
)

raw.set_montage(
    montage,
    on_missing='warn'
)


# =========================================================
# ADD MARKERS AS MNE ANNOTATIONS
# =========================================================

eeg_start_time = eeg_timestamps[0]

marker_timestamps = np.asarray(
    marker_stream['time_stamps'],
    dtype=float
)

marker_values = marker_stream['time_series']


annotation_onsets = []
annotation_durations = []
annotation_descriptions = []


for marker, timestamp in zip(
    marker_values,
    marker_timestamps
):

    marker_name = marker[0]

    # Marker time relative to start of EEG
    onset = timestamp - eeg_start_time

    # Ignore anything outside EEG recording
    if onset < 0:
        continue

    if onset > raw.times[-1]:
        continue

    annotation_onsets.append(onset)
    annotation_durations.append(0.0)
    annotation_descriptions.append(marker_name)


annotations = mne.Annotations(
    onset=annotation_onsets,
    duration=annotation_durations,
    description=annotation_descriptions
)

raw.set_annotations(annotations)


# =========================================================
# REPORT
# =========================================================

print("\n========================================")
print(" MNE OBJECT CREATED")
print("========================================")

print(raw)

print("\nChannels:")
print(raw.ch_names)

print("\nAnnotations:")

for annotation in raw.annotations:

    print(
        f"{annotation['onset']:8.3f}s  "
        f"{annotation['description']}"
    )


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE = XDF_FILE.with_name(
    XDF_FILE.stem + '_raw.fif'
)

raw.save(
    OUTPUT_FILE,
    overwrite=True
)

print()
print(f"Saved MNE file:")
print(OUTPUT_FILE)


# =========================================================
# OPEN MNE VIEWER
# =========================================================

print("\nOpening MNE viewer...")

raw.plot(
    duration=20,
    n_channels=5,
    scalings='auto',
    block=True
)
