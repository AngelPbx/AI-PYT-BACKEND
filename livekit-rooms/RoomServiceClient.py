import asyncio
import os
from livekit import api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get LiveKit credentials from environment variables
LIVEKIT_HOST = os.getenv('LIVEKIT_HOST')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

async def manage_rooms():
    if not all([LIVEKIT_HOST, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise ValueError("Missing required LiveKit configuration in environment variables")

    # Initialize LiveKit API with correct parameters
    livekit_api = api.LiveKitAPI(
        url=LIVEKIT_HOST,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )

    try:
        # Create a room
        print("Creating room...")
        room = await livekit_api.create_room(
            name="myroom",
            empty_timeout=10 * 60,  # 10 minutes
            max_participants=20
        )
        print(f"Room created: {room.name}")

        # List all rooms
        print("\nListing rooms...")
        rooms = await livekit_api.list_rooms()
        for r in rooms:
            print(f"- Room: {r.name}, Participants: {r.num_participants}")

        # Delete the room
        print("\nDeleting room...")
        await livekit_api.delete_room("myroom")
        print("Room deleted.")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Properly close the session using aclose()
        await livekit_api.aclose()

if __name__ == "__main__":
    asyncio.run(manage_rooms())
