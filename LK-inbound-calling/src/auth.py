import jwt
import time
from config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET

def generate_jwt():
    payload = {
        "iss": LIVEKIT_API_KEY,
        "exp": int(time.time()) + 3600,  # Expires in 1 hour
        "video": {
            "roomCreate": True,
            "roomList": True,
            "roomAdmin": True,
            "recording": True,
            "egress": True,
            "ingress": True,
        }
    }

    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
    return token
