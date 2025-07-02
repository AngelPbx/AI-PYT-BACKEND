import asyncio
import os
import time
import jwt
from livekit import rtc
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_WS_URL")  # Example: wss://5rlr6yk0zfz.sip.livekit.cloud
API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")


def generate_token(room_name, identity):
    payload = {
        "iss": API_KEY,
        "sub": identity,
        "roomJoin": True,
        "room": room_name,
        "exp": int(time.time()) + 3600,
        "video": {},
        "metadata": "agent"
    }
    token = jwt.encode(payload, API_SECRET, algorithm="HS256")
    return token


async def main():
    room_prefix = "call-"  # This should match your dispatch rule
    room_name = f"{room_prefix}test"  # Example room

    identity = "agent-bot-1"  # This is the agent identity
    token = generate_token(room_name, identity)

    # Connect to LiveKit
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"✅ Connected to room: {room_name}")

    # Set up event listeners
    @room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"👤 Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        print(f"❌ Participant disconnected: {participant.identity}")

    @room.on("track_subscribed")
    async def on_track_subscribed(track, publication, participant):
        print(f"🎧 Subscribed to track: {publication.track_sid} from {participant.identity}")

        if isinstance(track, rtc.AudioTrack):
            @track.on("data")
            async def on_audio_frame(frame):
                print(f"🔊 Received audio frame of {len(frame.samples)} samples")

                # (Optional) Play or process the audio here

    @room.on("disconnected")
    async def on_disconnected():
        print("🚪 Disconnected from room")

    try:
        while True:
            await asyncio.sleep(1)  # Keep the connection alive
    except KeyboardInterrupt:
        print("🛑 Disconnecting...")
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
