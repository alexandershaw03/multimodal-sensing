from pylsl import StreamInfo, StreamOutlet
import time


info = StreamInfo(
    name='ATS_MARKERS',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='string',
    source_id='ats_manual_markers'
)

outlet = StreamOutlet(info)

print()
print("========================================")
print(" ATS MARKER STREAM RUNNING")
print("========================================")
print()
print("Controls:")
print("  R = REST")
print("  L = LEFT_ARM")
print("  G = RIGHT_ARM")
print("  F = FORWARD_REACH")
print("  B = BLINK")
print("  S = STOP / END")
print("  Q = QUIT")
print()

while True:

    command = input("Marker > ").strip().upper()

    marker = None

    if command == 'R':
        marker = 'REST'

    elif command == 'L':
        marker = 'LEFT_ARM'

    elif command == 'G':
        marker = 'RIGHT_ARM'

    elif command == 'F':
        marker = 'FORWARD_REACH'

    elif command == 'B':
        marker = 'BLINK'

    elif command == 'S':
        marker = 'STOP'

    elif command == 'Q':
        print("Marker stream stopped.")
        break

    else:
        print("Unknown command.")
        continue

    outlet.push_sample([marker])

    print(f"Sent: {marker}")
