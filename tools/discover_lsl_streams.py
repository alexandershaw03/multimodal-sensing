from pylsl import resolve_streams

print("Searching for LSL streams...\n")

streams = resolve_streams()

for stream in streams:
    print(
        f"Name: {stream.name()} | "
        f"Type: {stream.type()} | "
        f"Channels: {stream.channel_count()} | "
        f"Rate: {stream.nominal_srate()} Hz"
    )
