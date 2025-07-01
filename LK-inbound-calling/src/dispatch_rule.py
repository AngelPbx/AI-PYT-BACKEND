from livekit.protocol.sip import CreateSIPDispatchRuleRequest, SIPDispatchRule, DispatchRuleDirect

async def create_dispatch_rule(client, trunk_id, room_name):
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
