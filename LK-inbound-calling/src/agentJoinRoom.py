from livekit import rtc
from auth import generate_jwt
from config import LIVEKIT_WS_URL
import asyncio

async def join_room():
    token = generate_jwt()
    room = rtc.Room()
    await room.connect(LIVEKIT_WS_URL, token)
    print(f"Connected to room: test-room")
    # Handle incoming audio/video tracks
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"Track subscribed: {track.kind} from {participant.identity}")
    await room.wait_for_disconnect()

if __name__ == "__main__":
    asyncio.run(join_room())