from livekit import api
from auth import generate_jwt
from config import LIVEKIT_WS_URL
import asyncio

async def create_trunk_and_rule():
    token = generate_jwt()

    # Connect to LiveKit gRPC using WS URL
    client = api.RoomServiceClient(LIVEKIT_WS_URL, token)

    # Create Inbound SIP Trunk
    trunk = await client.sip.create_sip_inbound_trunk({
        "trunk": {
            "name": "Test Trunk",
            "numbers": ["+18333659442"],
            "krisp_enabled": True
        }
    })
    print("Created Trunk:", trunk)

    # Create Dispatch Rule
    trunk_id = trunk.trunk.trunk_id
    rule = await client.sip.create_sip_dispatch_rule({
        "rule": {
            "dispatch_rule_direct": {
                "room_name": "test-room"
            }
        },
        "trunk_ids": [trunk_id]
    })
    print("Created Dispatch Rule:", rule)

asyncio.run(create_trunk_and_rule())
