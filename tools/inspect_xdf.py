import pyxdf
from pathlib import Path


# ---------------------------------------------------------
# CHANGE THIS TO YOUR XDF FILE
# ---------------------------------------------------------

# Change "xxx"'s to actual file path
XDF_FILE = Path(
    r" xxx "
)


print("\n========================================")
print(" ATS XDF INSPECTOR")
print("========================================\n")

print(f"Loading:\n{XDF_FILE}\n")

streams, header = pyxdf.load_xdf(str(XDF_FILE))

print(f"Found {len(streams)} stream(s).\n")


for i, stream in enumerate(streams, start=1):

    info = stream["info"]

    name = info["name"][0]
    stream_type = info["type"][0]

    samples = stream["time_series"]
    timestamps = stream["time_stamps"]

    print("----------------------------------------")
    print(f"STREAM {i}")
    print("----------------------------------------")

    print(f"Name:        {name}")
    print(f"Type:        {stream_type}")
    print(f"Samples:     {len(samples)}")

    if len(timestamps) > 0:
        duration = timestamps[-1] - timestamps[0]

        print(f"Duration:    {duration:.3f} seconds")

        if duration > 0 and len(samples) > 1:
            measured_rate = (len(samples) - 1) / duration
            print(f"Actual rate: {measured_rate:.2f} Hz")

    print()


print("========================================")
print(" STREAM CONTENT CHECK")
print("========================================\n")


for stream in streams:

    name = stream["info"]["name"][0]

    # -----------------------------------------------------
    # EEG
    # -----------------------------------------------------

    if name == "ATS_EEG_RAW":

        eeg = stream["time_series"]

        print("ATS_EEG_RAW:")
        print(f"  Shape: {eeg.shape}")

        if len(eeg) > 0:
            print("  First EEG sample:")
            print(f"    AF3 = {eeg[0][0]}")
            print(f"    T7  = {eeg[0][1]}")
            print(f"    Pz  = {eeg[0][2]}")
            print(f"    T8  = {eeg[0][3]}")
            print(f"    AF4 = {eeg[0][4]}")

        print()

    # -----------------------------------------------------
    # MARKERS
    # -----------------------------------------------------

    elif name == "ATS_MARKERS":

        markers = stream["time_series"]
        timestamps = stream["time_stamps"]

        print("ATS_MARKERS:")

        if len(markers) == 0:
            print("  No markers found.")

        else:

            first_time = timestamps[0]

            for marker, timestamp in zip(markers, timestamps):

                relative_time = timestamp - first_time

                print(
                    f"  +{relative_time:8.3f}s"
                    f"   {marker[0]}"
                )

        print()


print("========================================")
print(" DONE")
print("========================================")
