import asyncio
import time
import jwt
import json
from aiohttp import ClientSession

# LiveKit credentials
LIVEKIT_URL="https://trial-agent-m3zdvur5.livekit.cloud"
API_KEY="APIGp7PPyuhzxA6"
API_SECRET="MhfVPdp7vSFJelweQlXJfJS6rJs4OqjL6qfecGUEjuTD"

def generate_admin_token():
    now = int(time.time()) - 3000
    payload = {
        "iss": API_KEY,
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
    return jwt.encode(payload, API_SECRET, algorithm="HS256")

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
        "name": "ankit",
        "empty_timeout": 600,
        "max_participants": 10,
         "metadata": json.dumps({
            "kb_id": "df639e6aede94487"  # 👈 Pass your dynamic KB ID here
        })
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
                pretty_print("Current rooms", data)
            else:
                print(f"\n Error listing rooms ({resp.status}):\n{await resp.text()}")

if __name__ == "__main__":
    asyncio.run(list_rooms())
    # asyncio.run(create_room())
    
    #in this code the room is created and listed using the LiveKit API. what i want to do is whenever a participant joins the room, auto