import os
import asyncio
import jwt
import time
from dotenv import load_dotenv
from livekit import agents

def generate_agent_token():
    api_key = os.getenv('LIVEKIT_API_KEY')
    api_secret = os.getenv('LIVEKIT_API_SECRET')
    
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": "agent-id",
        "iat": now,
        "exp": now + 3600,
        "aud": "livekit.agent",
        "video": {
            "room": "*",
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True
        }
    }
    return jwt.encode(payload, api_secret, algorithm='HS256')

async def entrypoint(ctx: agents.JobContext):
    print("[INFO] Agent starting...")
    await ctx.connect()
    print("[INFO] Connected to LiveKit")
    # ... rest of your agent logic ...

if __name__ == "__main__":
    load_dotenv()
    
    token = generate_agent_token()
    options = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=os.getenv('LIVEKIT_HOST'),
        api_key=os.getenv('LIVEKIT_API_KEY'),
        api_secret=os.getenv('LIVEKIT_API_SECRET')
    )
    
    agents.cli.run_app(options)