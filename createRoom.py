import asyncio
import time
import jwt
from aiohttp import ClientSession

# LiveKit credentials
API_KEY = "APIGCNVJPqywiDJ"
API_SECRET = "8ek56lzz7m3eR0Htipt7wyDIhy6YSJAlYYdOBw4rwe7A"
LIVEKIT_URL = "https://create-agent-7nehkvvu.livekit.cloud"

def generate_admin_token():
    now = int(time.time()) - 3000
    payload = {
        "iss": "APIGCNVJPqywiDJ",
        "aud": "livekit",
        "iat": now,
        "exp": now + 3600,  # Token valid for 1 hour
        "video": {
            "roomCreate": True
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
                print("✅ Room created successfully:", data)
            else:
                print(f"❌ Error ({resp.status}): {await resp.text()}")

if __name__ == "__main__":
    asyncio.run(create_room())
