from livekit import api
from livekit.protocol.sip import CreateSIPDispatchRuleRequest, SIPDispatchRule, DispatchRuleDirect
from auth import generate_jwt
from config import LIVEKIT_WS_URL
import asyncio

async def create_or_verify_dispatch_rule(client, trunk_id, room_name):
    # Create dispatch rule
    rule = SIPDispatchRule(
        dispatch_rule_direct=DispatchRuleDirect(room_name=room_name)
    )
    req = CreateSIPDispatchRuleRequest(
        rule=rule,
        trunk_ids=[trunk_id],
        hide_phone_number=False
    )
    result = await client.sip.create_sip_dispatch_rule(req)
    print(f"Dispatch rule created: {result}")
    return result

async def main():
    # Generate JWT
    token = generate_jwt()
    # Connect to LiveKit
    client = api.RoomServiceClient(LIVEKIT_WS_URL, token)
    # Use existing trunk SID and room name
    trunk_id = "TKab958946719f99ce2"
    room_name = "test-room"
    # Create or update dispatch rule
    await create_or_verify_dispatch_rule(client, trunk_id, room_name)

if __name__ == "__main__":
    asyncio.run(main())