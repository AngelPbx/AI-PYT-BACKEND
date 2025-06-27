import asyncio
import time
import jwt
import json
from aiohttp import ClientSession
import os
# LiveKit credentials
LIVEKIT_URL = os.getenv('LIVEKIT_URL') # Ensure this env var is set
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY') # Ensure this env var is set
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

def generate_admin_token():
    now = int(time.time()) - 3000
    payload = {
        "iss": LIVEKIT_API_KEY,
        "aud": "livekit",
        "sub": "admin",
        "iat": now - 3000,
        "exp": now + 3600,
        "video": {
            "roomCreate": True,
            "roomList": True
        }
    }
    # if room_name:
    #     payload["sub"] = identity  # ✅ Include only if needed
    #     payload["room"] = room_name  # ✅ Include only if needed
    return jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")

def pretty_print(title, data):
    print(f"\n==> {title}")
    print(json.dumps(data, indent=2))

async def create_room():
    token = generate_admin_token()
    url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/CreateRoom"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "name": "new-ankit-6000",
        "empty_timeout": 6000,
        "max_participants": 10,
         "metadata": json.dumps({"kb_id": "df639e6aede94487","model_stt": "nova-3","lang_stt": "multi","model_llm": "gpt-4o-mini","model_tts": "gpt-4o-mini-tts","voice_tts": "nova", "AGENT_PERSONA": "You are a Doctor AI assistant."})
    }
    #create room
    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                pretty_print("Room created successfully", data)
            else:
                print(f"\n Error creating room ({resp.status}):\n{await resp.text()}")

async def list_rooms():
    token = generate_admin_token()
    url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/ListRooms"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    async with ClientSession() as session:
        async with session.post(url, headers=headers, json={}) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("📋 Current rooms:", data)
            else:
                print(f"❌ Error listing rooms ({resp.status}): {await resp.text()}")

if __name__ == "__main__":
    asyncio.run(list_rooms())
    # asyncio.run(create_room())
    
    #in this code the room is created and listed using the LiveKit API. what i want to do is whenever a participant joins the room, auto
