"""
A live terminal monitor, for ATS_EEG_LSL stream.
Resolves ATS_EEG_RAW, opens LSL inlet and displays representative, live, five-channel EEG samples (without absolutely flooding the terminal lol).

Usage
-----
python tools/read_eeg_stream.py

Optional:
python tools/read_eeg_stream.py --print-rate 10
python tools/read_eeg_stream.py --timeout 10
"""

from __future__ import annotations

import argparse
import time

from pylsl import StreamInlet, resolve_byprop


DEFAULT_STREAM_NAME = "ATS_EEG_RAW"

EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)


def parse_args() -> argparse.Namespace:
    """Parse live-reader config"""

    parser = argparse.ArgumentParser(
        description=(
            "Read and display live ATS EEG samples from Lab Streaming Layer."
        )
    )

    parser.add_argument(
        "--name",
        default=DEFAULT_STREAM_NAME,
        help=(
            "LSL stream name "
            f"(default: {DEFAULT_STREAM_NAME})."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help=(
            "Seconds to wait while resolving the stream (default: 5)."
        ),
    )

    parser.add_argument(
        "--print-rate",
        type=float,
        default=5.0,
        help=(
            "Maximum terminal updates per second (default: 5)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be greater than zero."
        )

    if args.print_rate <= 0:
        raise ValueError(
            "--print-rate must be greater than zero."
        )

    print()
    print("========================================")
    print(" ATS LIVE EEG MONITOR")
    print("========================================")
    print()

    print(
        f"Searching for {args.name}..."
    )

    streams = resolve_byprop(
        "name",
        args.name,
        timeout=args.timeout,
    )

    if not streams:
        raise RuntimeError(
            f"LSL stream '{args.name}' was not found.\n"
            "Make sure acquisition/eeg_lsl_stream.py is running and showing stream."
        )

    if len(streams) > 1:
        print(
            f"Warning: {len(streams)} matching streams found; "
            "using the first."
        )

    stream_info = streams[0]

    print()
    print("Stream found:")
    print(
        f"  Name:       {stream_info.name()}"
    )
    print(
        f"  Type:       {stream_info.type()}"
    )
    print(
        f"  Channels:   {stream_info.channel_count()}"
    )
    print(
        f"  Nominal Fs: {stream_info.nominal_srate():.3f} Hz"
    )

    if (
        stream_info.channel_count()
        !=
        len(EEG_CHANNELS)
    ):
        print()
        print(
            "WARNING: stream channel count does not match the expected five-channel ATS EEG layout."
        )

    print()
    print("Connecting...")

    inlet = StreamInlet(
        stream_info
    )

    print("Connected.")
    print()

    print(
        "".join(
            f"{channel:>12}"
            for channel in EEG_CHANNELS
        )
    )

    print(
        "-" * (
            len(EEG_CHANNELS)
            * 12
        )
    )

    print_interval = (
        1.0
        /
        args.print_rate
    )

    last_print = 0.0
    sample_count = 0
    start_time = time.monotonic()

    try:
        while True:
            sample, _timestamp = inlet.pull_sample(
                timeout=1.0
            )

            if sample is None:
                print(
                    "Waiting for EEG samples..."
                )
                continue

            sample_count += 1

            if len(sample) < len(
                EEG_CHANNELS
            ):
                print(
                    f"WARNING: received only "
                    f"{len(sample)} channels."
                )
                continue

            now = time.monotonic()

            if (
                now
                -
                last_print
                <
                print_interval
            ):
                continue

            print(
                "".join(
                    f"{float(value):12.2f}"
                    for value in sample[
                        :len(EEG_CHANNELS)
                    ]
                )
            )

            last_print = now

    except KeyboardInterrupt:
        elapsed = max(
            time.monotonic()
            -
            start_time,
            1e-9,
        )

        observed_delivery_rate = (
            sample_count
            /
            elapsed
        )

        print()
        print()
        print("Stream monitor stopped.")
        print(
            f"Samples received: "
            f"{sample_count}"
        )
        print(
            f"Approx. receive rate: "
            f"{observed_delivery_rate:.2f} samples/s"
        )


if __name__ == "__main__":
    main()
