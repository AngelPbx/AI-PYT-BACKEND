import asyncio
from livekit.api import CreateRoomRequest, ListRoomsRequest
from livekit import api
from livekit.protocol.sip import ListSIPInboundTrunkRequest
LIVEKIT_URL="https://bestagent-8alh62on.livekit.cloud"
LIVEKIT_API_KEY="APIfie4gtLsymHR"
LIVEKIT_API_SECRET="70RUoZ76qtzC2PeRGWMBAFWlbIxvReLeUEbMiJPxZ2aB"
async def main():
  async with api.LiveKitAPI(
    
      url=LIVEKIT_URL,
      api_key=LIVEKIT_API_KEY,
      api_secret=LIVEKIT_API_SECRET,
  ) as lkapi:
    room = await lkapi.room.create_room(CreateRoomRequest(
    name="myroom",
    empty_timeout=10 * 60,
    max_participants=20,
  ))
    # rooms = await lkapi.room.list_rooms(ListRoomsRequest())
    # print("Current rooms:",rooms)

asyncio.run(main())