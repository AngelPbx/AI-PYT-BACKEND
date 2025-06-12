import asyncio
import os
from dotenv import load_dotenv
from livekit.api import LiveKitAPI, CreateRoomRequest
from livekit import api
load_dotenv()

async def main():
    print("LIVEKIT_URL:", os.getenv("LIVEKIT_URL"))
    print("LIVEKIT_API_KEY:", os.getenv("LIVEKIT_API_KEY"))
    print("LIVEKIT_API_SECRET:", os.getenv("LIVEKIT_API_SECRET"))

    # async with LiveKitAPI() as lkapi:
    #     response = await lkapi.room.create_room(CreateRoomRequest(name="ankit"))  # Await the coroutine
    #     print("Response:", response)
    async with LiveKitAPI() as lkapi:
        try:
            response = await lkapi.room.create_room(CreateRoomRequest(name="ankit"))
            print("Response:", response.text)
        except Exception as e:
            print("Error:", e)



    

#     lkapi = api.LiveKitAPI(
#         "https://natty-gz614tko.livekit.cloud",  # Replace with your LiveKit server URL
#         api_key="APIU7upDtrpdrEK",  # Replace with your API key
#         api_secret="LWfMIqkaYAf7MgfqHnmCGkdt2jjPWeTTGxFHmvJJJHND"  # Replace with your API secret
#     )
    
#     room_info = await lkapi.room.create_room(
#        api.CreateRoomRequest(name="my-room", empty_timeout=10 * 60, max_participants=20)
#    )
    # print("---------------",room_info,"--------------")
    await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
