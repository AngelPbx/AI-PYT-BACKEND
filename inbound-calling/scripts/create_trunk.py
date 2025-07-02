import asyncio
import json
from livekit import api
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    livekit_api = None
    try:
        # Load config
        config_path = Path(__file__).parent / "inbound-trunk.json"
        with open(config_path, 'r') as f:
            data = json.load(f)

        trunk_data = data["trunk"]
        if not trunk_data:
            print("No trunk data in JSON")
            return

        livekit_api = api.LiveKitAPI(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            url=os.getenv("LIVEKIT_URL").replace("wss://", "https://")
        )

        trunk = api.SIPInboundTrunkInfo(
            name=trunk_data['name'],
            numbers=trunk_data['number'],
            auth_username=trunk_data['username'],
            auth_password=trunk_data['password'],
            krisp_enabled=True
        )

        request = api.CreateSIPInboundTrunkRequest(trunk=trunk)

        result = await livekit_api.sip.create_sip_inbound_trunk(request)

        print(f"✅ Trunk Created: {result}")

    except Exception as e:
        print(f"❌ Error creating trunk: {e}")
    finally:
        if livekit_api:
            await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(main())
