import jwt
import time
import requests
import os
from dotenv import load_dotenv

# Fetch credentials from environment
API_KEY = "APIU7upDtrpdrEK"
API_SECRET = "LWfMIqkaYAf7MgfqHnmCGkdt2jjPWeTTGxFHmvJJJHND"

# LiveKit endpoint  
LIVEKIT_URL = "https://natty-gz614tko.livekit.cloud/twirp/livekit.RoomService/CreateRoom"


def generate_jwt(api_key: str, api_secret: str, ttl: int = 3600) -> str:
    payload = {
        "iss": api_key,
        "exp": int(time.time()) + ttl,
        "roomCreate": True 
    }
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token



def create_room(room_name: str):
    """Create a LiveKit room using the CreateRoom endpoint."""
    token = generate_jwt(API_KEY, API_SECRET)
    
    print("Generated JWT token:", token)
    if not token:
        print("==> Failed to generate JWT token")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    data = {
        "name": room_name
    }

    response = requests.post(LIVEKIT_URL, headers=headers, json=data)

    if response.status_code == 200:
        print("==> Room created:", response.json())
    else:
        print("==> Failed to create room:", response.status_code, response.text)


# Example usage
if __name__ == "__main__":
    create_room("demo-room")
