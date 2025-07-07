from livekit.protocol.sip import SIPInboundTrunkInfo, CreateSIPInboundTrunkRequest

async def create_inbound_trunk(client, number, allowed_numbers=[]):
    trunk = SIPInboundTrunkInfo(
        name="Twilio Trunk",
        numbers=[number],
        allowed_numbers=allowed_numbers,
        krisp_enabled=True,
    )
    req = CreateSIPInboundTrunkRequest(trunk=trunk)
    result = await client.sip.create_sip_inbound_trunk(req)
    print(f"Inbound trunk created: {result}")
    return result
