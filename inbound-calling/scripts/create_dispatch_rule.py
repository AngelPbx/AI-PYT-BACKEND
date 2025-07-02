import os
import time
import jwt
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def generate_jwt():
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    payload = {
        "iss": api_key,
        "exp": int(time.time()) + 600,
        "video": {},
        "metadata": "sip"
    }
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    return token


def create_dispatch_rule():
    try:
        config_path = Path(__file__).parent / "dispatch-rule-sales.json"
        with open(config_path, 'r') as f:
            body = json.load(f)

        url = os.getenv("LIVEKIT_URL").replace("wss://", "https://") + "/twirp/livekit.SIPSignal/CreateSIPDispatchRule"
        token = generate_jwt()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        print(f"🚀 Sending request to {url} ...")
        response = requests.post(url, headers=headers, json=body)

        if response.ok:
            print("✅ Dispatch Rule Created Successfully!")
            try:
                data = response.json()
                print(json.dumps(data, indent=4))
            except ValueError:
                print("ℹ️ Success but empty or non-JSON response. Check the dashboard.")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Exception occurred: {e}")


if __name__ == "__main__":
    create_dispatch_rule()
