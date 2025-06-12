import asyncio
import time
import jwt
from aiohttp import ClientSession

# LiveKit credentials
API_KEY = "APIFTbB7ZDuDmwL"
API_SECRET = "gE63Bpx7amNAnG77voI9QfeEX4vCWFrPNIQTceKGStRB"
LIVEKIT_URL = "https://bestagent-8alh62on.livekit.cloud"

def generate_admin_token():
    now = int(time.time()) - 3000
    payload = {
        "iss": API_KEY,
        "aud": "livekit",
        "iat": now,
        "exp": now + 3600,
        "video": {
            "roomCreate": True,
            "roomList": True  # ✅ Important for listing rooms
        }
    }
    return jwt.encode(payload, API_SECRET, algorithm="HS256")

async def create_room():
    token = generate_admin_token()
    url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/CreateRoom"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "name": "myroom",
        "empty_timeout": 600,
        "max_participants": 10
    }

    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("Room created successfully:", data)
            else:
                print(f"❌ Error creating room ({resp.status}): {await resp.text()}")

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
    # asyncio.run(create_room())  # You can toggle this
