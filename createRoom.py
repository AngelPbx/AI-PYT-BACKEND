import asyncio
import time
import jwt
from aiohttp import ClientSession

# LiveKit credentials
# API_KEY = "APIFTbB7ZDuDmwL"
# API_SECRET = "gE63Bpx7amNAnG77voI9QfeEX4vCWFrPNIQTceKGStRB"
# LIVEKIT_URL = "https://bestagent-8alh62on.livekit.cloud"

API_KEY = "APIU7upDtrpdrEK"
API_SECRET = "LWfMIqkaYAf7MgfqHnmCGkdt2jjPWeTTGxFHmvJJJHND"
LIVEKIT_URL = "https://natty-gz614tko.livekit.cloud"
def generate_participant_token(room_name, identity):
    now = int(time.time())
    payload = {
        "iss": API_KEY,
        "sub": identity,
        "aud": "livekit",
        "room": room_name,
        "iat": now,
        "exp": now + 3600,
        "video": {
            "roomJoin": True
        }
    }
    return jwt.encode(payload, API_SECRET, algorithm="HS256")
def generate_admin_token(room_name=None,identity=None):
    now = int(time.time())
    print(now,"Generating admin token with room_name:", room_name, "and identity:", identity)
    payload = {
        "iss": API_KEY,
        "aud": "livekit",
        "sub": "admin",
        "iat": now - 3000,
        "exp": now + 3600,
        "video": {
            "roomCreate": True,
            "roomList": True,
            "roomJoin": True,
            "roomAdmin": True,
            "roomRecord": True
        }
    }
    if room_name:
        payload["sub"] = identity  # ✅ Include only if needed
        payload["room"] = room_name  # ✅ Include only if needed
    return jwt.encode(payload, API_SECRET, algorithm="HS256")

async def create_room():
    token = generate_admin_token()
    url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/CreateRoom"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "name": "ankit",
        "empty_timeout": 600,
        "max_participants": 10
    }
    #create room
    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✅ Room created successfully:", data)
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

async def list_participants(room_name: str, identity: str):
    token = generate_participant_token(room_name,identity)
    url = f"{LIVEKIT_URL}/twirp/livekit.RoomService/ListParticipants"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "room": room_name
    }

    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"👥 Participants in room '{room_name}':", data)
            else:
                print(f"❌ Error listing participants ({resp.status}): {await resp.text()}")


if __name__ == "__main__":
    asyncio.run(create_room())
    # asyncio.run(list_rooms())
    # asyncio.run(list_participants("ankit","admin"))  # Specify the room name and identity
      # You can toggle this
