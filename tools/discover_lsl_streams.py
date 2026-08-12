"""
Discovers LSL streams on the local network. 
Mostly used to check whether acquisition, marker, experiment and vision processes are correctly displaying their LSL outlets.

Usage
-----
python tools/discover_lsl_streams.py

Only show ATS streams:
python tools/discover_lsl_streams.py --ats-only
"""

from __future__ import annotations

import argparse

from pylsl import resolve_streams


ATS_PREFIX = "ATS_"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Discover LSL streams currently visible on local network."
        )
    )

    parser.add_argument(
        "--ats-only",
        action="store_true",
        help="Only display streams whose names begin with ATS_.",
    )

    return parser.parse_args()


def discover_streams(
    ats_only: bool = False,
):
    """Resolve (and optionally filter) visible LSL streams."""

    streams = resolve_streams()

    if ats_only:
        streams = [
            stream
            for stream in streams
            if stream.name().startswith(ATS_PREFIX)
        ]

    return sorted(
        streams,
        key=lambda stream: (
            stream.hostname(),
            stream.name(),
            stream.type(),
        ),
    )


def print_streams(streams) -> None:
    """Print discovered streams in a table."""

    print()
    print("========================================")
    print(" LSL STREAM DISCOVERY")
    print("========================================")
    print()

    if not streams:
        print("No matching LSL streams found.")
        print()
        return

    print(
        f"{'NAME':<24}"
        f"{'TYPE':<14}"
        f"{'CH':>5}"
        f"{'RATE (Hz)':>13}  "
        f"{'HOST'}"
    )

    print("-" * 78)

    for stream in streams:
        nominal_rate = float(
            stream.nominal_srate()
        )

        rate_text = (
            f"{nominal_rate:.2f}"
            if nominal_rate > 0
            else "irregular"
        )

            hostname = (
                stream.hostname()
                or "unknown"
            )

        print(
            f"{stream.name():<24}"
            f"{stream.type():<14}"
            f"{stream.channel_count():>5}"
            f"{rate_text:>13}  "
            f"{hostname}"
        )

    print()
    print(
        f"{len(streams)} stream"
        f"{'' if len(streams) == 1 else 's'} found."
    )
    print()


def main() -> None:
    args = parse_args()

    print("Searching for LSL streams...")

    streams = discover_streams(
        ats_only=args.ats_only
    )

    print_streams(
        streams
    )


if __name__ == "__main__":
    main()
