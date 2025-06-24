import asyncio
from livekit.api import LiveKitAPI, CreateRoomRequest
import os


async def create_room():
    LIVEKIT_URL = os.getenv('LIVEKIT_URL')
    LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
    LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

    # ✅ Move inside async block
    lkapi = LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    room_service = lkapi.room
    room_name = "my_new_room"

    room_info = await room_service.create_room(
        CreateRoomRequest(
            name=room_name,
            empty_timeout=10 * 60,
            max_participants=20
        )
    )
    print(f"✅ Room created: {room_info.name}, SID: {room_info.sid}")


# ✅ Run inside async context
asyncio.run(create_room())
