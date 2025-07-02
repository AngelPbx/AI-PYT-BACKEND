import asyncio
import json
from livekit import api
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    try:
        config_path = Path(__file__).parent / "dispatch-rule.json"
        with open(config_path, 'r') as f:
            data = json.load(f)

        rule_data = data.get('rule')
        if not rule_data:
            print("No dispatch rule data in JSON")
            return

        livekit_api = api.LiveKitAPI(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            url=os.getenv("LIVEKIT_URL").replace("wss://", "https://")
        )

        request = api.CreateSIPDispatchRuleRequest(
            rule=api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                    room_prefix=rule_data['room_prefix']
                )
            ),
            room_config=api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=rule_data['agent_name'],
                        metadata=rule_data.get('metadata', "")
                    )
                ]
            )
        )

        result = await livekit_api.sip.create_sip_dispatch_rule(request)

        print(f"Dispatch Rule Created: {result}")

    except Exception as e:
        print(f"Error creating dispatch rule: {e}")
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(main())