"""
Manual LSL marker publisher.

Publishes timestamped experimental event markers to the ATS_MARKERS LSL stream. 
Intended for manual annotation during multimodal recording, including rest periods, arm movements, blinks and experiment termination.

Output stream
-------------
Name:
    ATS_MARKERS

Type:
    Markers

Channel count:
    1

Markers:
    REST
    LEFT_ARM
    RIGHT_ARM
    FORWARD_REACH
    BLINK
    STOP
"""

from __future__ import annotations

from pylsl import StreamInfo, StreamOutlet, local_clock


STREAM_NAME = "ATS_MARKERS"
STREAM_TYPE = "Markers"
SOURCE_ID = "ats_manual_markers"

COMMANDS = {
    "R": "REST",
    "L": "LEFT_ARM",
    "G": "RIGHT_ARM",
    "F": "FORWARD_REACH",
    "B": "BLINK",
    "S": "STOP",
}


def create_marker_outlet() -> StreamOutlet:
    """Create manual marker LSL outlet."""

    info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=SOURCE_ID,
    )

    info.desc().append_child_value(
        "manufacturer",
        "ATS",
    )

    info.desc().append_child_value(
        "purpose",
        "Manual behavioural and experimental event annotation",
    )

    return StreamOutlet(info)


def print_controls() -> None:
    """Display available manual marker commands."""

    print()
    print("========================================")
    print(" ATS MANUAL MARKER STREAM")
    print("========================================")
    print()
    print(f"LSL stream: {STREAM_NAME}")
    print()
    print("Controls:")
    print(" R = REST")
    print(" L = LEFT_ARM")
    print(" G = RIGHT_ARM")
    print(" F = FORWARD_REACH")
    print(" B = BLINK")
    print(" S = STOP / END")
    print(" Q = QUIT")
    print()


def main() -> None:
    """Run the interactive manual marker publisher."""

    outlet = create_marker_outlet()

    print_controls()

    try:
        while True:
            command = input("Marker > ").strip().upper()

            if command == "Q":
                print("Marker stream stopped.")
                break

            marker = COMMANDS.get(command)

            if marker is None:
                print(
                    "Unknown command. "
                    f"Valid commands: {', '.join(COMMANDS)}, Q"
                )
                continue

            timestamp = local_clock()

            outlet.push_sample(
                [marker],
                timestamp=timestamp,
            )

            print(
                f"Sent: {marker:<14} "
                f"LSL time: {timestamp:.6f}"
            )

    except (KeyboardInterrupt, EOFError):
        print("\nMarker stream stopped.")


if __name__ == "__main__":
    main()
