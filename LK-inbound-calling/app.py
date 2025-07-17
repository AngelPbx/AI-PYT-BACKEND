from flask import Flask, Response, request
import asyncio
from src.auth import generate_jwt
from src.config import LIVEKIT_WS_URL
from livekit.protocol.sip import CreateSIPDispatchRuleRequest, SIPDispatchRule, DispatchRuleDirect
from livekit import api

app = Flask(__name__)

@app.route("/voice", methods=['POST'])
def voice():
    print("📞 Incoming call received:", request.form)
    response = """
    <?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Dial>
            <Sip>sip:2tfpk869du5.sip.livekit.cloud</Sip>
        </Dial>
    </Response>
    """
    return Response(response, mimetype='text/xml')

@app.route("/", methods=["GET"])
def home():
    return "Flask Server is running!"

async def create_dispatch_rule():
    token = generate_jwt()
    client = api.RoomServiceClient(LIVEKIT_WS_URL, token)
    trunk_id = "TKababb2b958dfb94676419f99ce2893ee"
    room_name = "test-room"
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

def run_flask():
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # Create dispatch rule
    asyncio.run(create_dispatch_rule())
    # Start Flask app
    run_flask()
