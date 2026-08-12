"""
Emotiv Cortex -> Lab Streaming Layer EEG bridge.

Connects to the local Emotiv Cortex service, subscribes to raw EEG from an Emotiv Insight headset, and publishes the five EEG channels as an LSL stream.

Output stream
-------------
Name:
    ATS_EEG_RAW

Type:
    EEG

Channels:
    AF3, T7, Pz, T8, AF4

Nominal sample rate:
    128 Hz

Credentials loaded from environment variables or from a local `.env` file in the repository root:

    EMOTIV_CLIENT_ID
    EMOTIV_CLIENT_SECRET

"""

from __future__ import annotations

import argparse
import json
import os
import ssl
from pathlib import Path
from typing import Any

import websocket
from dotenv import load_dotenv
from pylsl import StreamInfo, StreamOutlet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORTEX_URL = "wss://localhost:6868"

EEG_CHANNELS = (
    "AF3",
    "T7",
    "Pz",
    "T8",
    "AF4",
)

NOMINAL_SAMPLE_RATE = 128.0

LSL_STREAM_NAME = "ATS_EEG_RAW"
LSL_STREAM_TYPE = "EEG"
LSL_SOURCE_ID = "emotiv_insight_ats"


# Load local .env file from the repository root, if it exists.
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


class CortexLSLBridge:
    """
    Bridge raw Emotiv Cortex EEG samples into Lab Streaming Layer.

    The Cortex API is accessed directly over its local JSON-RPC WebSocket interface, avoiding dependency on the Emotiv example `cortex.py`wrapper.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        headset_id: str | None = None,
        debug: bool = False,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.requested_headset_id = headset_id
        self.debug = debug

        self.ws = None
        self.request_id = 1

        self.cortex_token: str | None = None
        self.headset_id: str | None = None
        self.session_id: str | None = None

        self.channel_indices: list[int] | None = None

        self.eeg_outlet = self._create_lsl_outlet()

    # ------------------------------------------------------------------
    # LSL
    # ------------------------------------------------------------------

    @staticmethod
    def _create_lsl_outlet() -> StreamOutlet:
        """Create the five-channel raw EEG LSL outlet."""

        info = StreamInfo(
            name=LSL_STREAM_NAME,
            type=LSL_STREAM_TYPE,
            channel_count=len(EEG_CHANNELS),
            nominal_srate=NOMINAL_SAMPLE_RATE,
            channel_format="float32",
            source_id=LSL_SOURCE_ID,
        )

        channels = info.desc().append_child("channels")

        for label in EEG_CHANNELS:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("unit", "microvolts")
            channel.append_child_value("type", "EEG")

        info.desc().append_child_value("manufacturer", "EMOTIV")
        info.desc().append_child_value("model", "Insight")

        return StreamOutlet(info)

    # ------------------------------------------------------------------
    # Cortex JSON-RPC
    # ------------------------------------------------------------------

    def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Send one JSON-RPC request to Cortex and wait for matching reply.
        """

        if self.ws is None:
            raise RuntimeError("Cortex WebSocket is not connected.")

        current_id = self.request_id
        self.request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": method,
            "params": params or {},
        }

        if self.debug:
            print(f"[Cortex] -> {method}")

        self.ws.send(json.dumps(payload))

        while True:
            message = json.loads(self.ws.recv())

            # Ignore asynchronous messages while waiting for RPC reply.
            if message.get("id") != current_id:
                continue

            if "error" in message:
                raise RuntimeError(
                    f"Cortex {method} failed: {message['error']}"
                )

            return message.get("result")

    def connect(self) -> None:
        """Connect to local Cortex WebSocket service."""

        print(f"[Cortex] Connecting to {CORTEX_URL} ...")

        self.ws = websocket.create_connection(
            CORTEX_URL,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            timeout=10,
        )

        print("[Cortex] WebSocket connected.")

    def request_access(self) -> None:
        """Request application access through Cortex."""

        result = self._send_request(
            "requestAccess",
            {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
            },
        )

        if isinstance(result, dict):
            granted = result.get("accessGranted")

            if granted is False:
                raise RuntimeError(
                    "Cortex access not granted. "
                    "Check Emotiv app permissions."
                )

        print("[Cortex] Application access granted.")

    def authorize(self) -> None:
        """Authorize in app and obtain Cortex token."""

        result = self._send_request(
            "authorize",
            {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
                "debit": 1,
            },
        )

        if not isinstance(result, dict) or "cortexToken" not in result:
            raise RuntimeError(
                "Cortex authorization succeeded but returned no token."
            )

        self.cortex_token = result["cortexToken"]

        print("[Cortex] Authorized.")

    def select_headset(self) -> None:
        """Find and select requested or available headset."""

        headsets = self._send_request("queryHeadsets")

        if not headsets:
            raise RuntimeError(
                "No Emotiv headset was found. "
                "Check the headset is powered on and visible in Emotiv app."
            )

        if self.requested_headset_id:
            matching = [
                headset
                for headset in headsets
                if headset.get("id") == self.requested_headset_id
            ]

            if not matching:
                available = [
                    headset.get("id", "unknown")
                    for headset in headsets
                ]

                raise RuntimeError(
                    f"Requested headset "
                    f"'{self.requested_headset_id}' was not found. "
                    f"Available headsets: {available}"
                )

            selected = matching[0]

        else:
            selected = headsets[0]

        self.headset_id = selected["id"]

        print(f"[Cortex] Using headset: {self.headset_id}")

    def create_session(self) -> None:
        """Create active Cortex session for selected headset."""

        if self.cortex_token is None or self.headset_id is None:
            raise RuntimeError(
                "Cortex must be authorized and headset selected before creating session."
            )

        result = self._send_request(
            "createSession",
            {
                "cortexToken": self.cortex_token,
                "headset": self.headset_id,
                "status": "active",
            },
        )

        if not isinstance(result, dict) or "id" not in result:
            raise RuntimeError(
                "Cortex did not return a valid session ID."
            )

        self.session_id = result["id"]

        print(f"[Cortex] Session created: {self.session_id}")

    def subscribe_eeg(self) -> None:
        """
        Subscribe to Cortex EEG stream and determine channel positions.
        """

        if self.cortex_token is None or self.session_id is None:
            raise RuntimeError(
                "Active Cortex session required before subscribing."
            )

        result = self._send_request(
            "subscribe",
            {
                "cortexToken": self.cortex_token,
                "session": self.session_id,
                "streams": ["eeg"],
            },
        )

        eeg_columns = None

        if isinstance(result, dict):
            for stream in result.get("success", []):
                if stream.get("streamName") == "eeg":
                    eeg_columns = stream.get("cols")
                    break

        if eeg_columns:
            try:
                self.channel_indices = [
                    eeg_columns.index(channel)
                    for channel in EEG_CHANNELS
                ]

                print(
                    "[Cortex] EEG channel mapping: "
                    + ", ".join(
                        f"{channel}={index}"
                        for channel, index in zip(
                            EEG_CHANNELS,
                            self.channel_indices,
                        )
                    )
                )

            except ValueError as exc:
                raise RuntimeError(
                    "Cortex EEG metadata did not contain all expected "
                    f"Insight channels. Returned columns: {eeg_columns}"
                ) from exc

        else:
            # Known Insight EEG layout used as fallback if Cortex does not provide stream-column metadata.
            self.channel_indices = [2, 3, 4, 5, 6]

            print(
                "[Cortex] Warning: no EEG column metadata returned; "
                "using standard Insight channel layout."
            )

        print("[Cortex] EEG subscription active.")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _publish_eeg(self, message: dict[str, Any]) -> None:
        """Extract one EEG sample from Cortex and push into LSL."""

        eeg_data = message.get("eeg")

        if eeg_data is None or self.channel_indices is None:
            return

        if max(self.channel_indices) >= len(eeg_data):
            if self.debug:
                print(
                    "[Cortex] Ignoring malformed EEG packet: "
                    f"received {len(eeg_data)} values."
                )
            return

        sample = [
            float(eeg_data[index])
            for index in self.channel_indices
        ]

        self.eeg_outlet.push_sample(sample)

    def stream(self) -> None:
        """Receive Cortex packets continuously and publish EEG to LSL."""

        if self.ws is None:
            raise RuntimeError("Cortex not connected.")

        # Once streaming starts, EEG packets should arrive continuously.
        self.ws.settimeout(None)

        print()
        print("========================================")
        print(" ATS EEG LSL STREAM ACTIVE")
        print(f" Stream:      {LSL_STREAM_NAME}")
        print(f" Channels:    {', '.join(EEG_CHANNELS)}")
        print(f" Nominal Fs:  {NOMINAL_SAMPLE_RATE:.0f} Hz")
        print("========================================")
        print()
        print("Press Ctrl+C to stop.")

        sample_count = 0

        while True:
            message = json.loads(self.ws.recv())

            if "eeg" not in message:
                continue

            self._publish_eeg(message)

            sample_count += 1

            if self.debug and sample_count % 128 == 0:
                print(
                    f"[LSL] Published {sample_count} EEG samples."
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialize Cortex and begin EEG-to-LSL streaming."""

        self.connect()
        self.request_access()
        self.authorize()
        self.select_headset()
        self.create_session()
        self.subscribe_eeg()
        self.stream()

    def close(self) -> None:
        """Close Cortex WebSocket connection."""

        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None

        print("\n[Cortex] Connection closed.")


def load_credentials() -> tuple[str, str]:
    """Read Cortex credentials from the environment."""

    client_id = os.getenv("EMOTIV_CLIENT_ID")
    client_secret = os.getenv("EMOTIV_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Emotiv Cortex credentials.\n"
            "Set EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET in a local .env file or environment variables."
        )

    return client_id, client_secret


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Stream raw Emotiv Insight EEG from Cortex to Lab Streaming Layer."
        )
    )

    parser.add_argument(
        "--headset-id",
        default=None,
        help=(
            "Optional Cortex headset ID. "
            "If omitted, first available headset is used."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional Cortex and LSL diagnostic information.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client_id, client_secret = load_credentials()

    bridge = CortexLSLBridge(
        client_id=client_id,
        client_secret=client_secret,
        headset_id=args.headset_id,
        debug=args.debug,
    )

    try:
        bridge.run()

    except KeyboardInterrupt:
        print("\nStopping EEG stream...")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
