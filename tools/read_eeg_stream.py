from pylsl import resolve_byprop, StreamInlet
import time


CHANNELS = ['AF3', 'T7', 'Pz', 'T8', 'AF4']


print("Searching for ATS_EEG_RAW...")

streams = resolve_byprop(
    'name',
    'ATS_EEG_RAW',
    timeout=5
)

if not streams:
    print("ERROR: ATS_EEG_RAW was not found.")
    print("Make sure ats_eeg_lsl.py is still running.")
    raise SystemExit

print("ATS_EEG_RAW found.")
print("Connecting...\n")

inlet = StreamInlet(streams[0])

print("Connected.")
print("Reading live EEG...\n")

print(
    f"{'AF3':>10} "
    f"{'T7':>10} "
    f"{'Pz':>10} "
    f"{'T8':>10} "
    f"{'AF4':>10}"
)

print("-" * 54)

last_print = 0

try:
    while True:

        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample is None:
            print("Waiting for EEG samples...")
            continue

        # Cortex is producing ~128 samples/s.
        # Only print ~5 times/s so terminal stays readable.
        now = time.time()

        if now - last_print >= 0.2:

            print(
                f"{sample[0]:10.2f} "
                f"{sample[1]:10.2f} "
                f"{sample[2]:10.2f} "
                f"{sample[3]:10.2f} "
                f"{sample[4]:10.2f}"
            )

            last_print = now

except KeyboardInterrupt:
    print("\nStopped.")
