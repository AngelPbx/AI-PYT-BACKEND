import asyncio

from livekit import api
from livekit.protocol.sip import ListSIPInboundTrunkRequest

async def main():
  livekit_api = api.LiveKitAPI()

  rules = await livekit_api.sip.list_sip_inbound_trunk(
    ListSIPInboundTrunkRequest()
  )
  print(f"{rules}")

  await livekit_api.aclose()

asyncio.run(main())