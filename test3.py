# from dotenv import load_dotenv
# from livekit import agents
# from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
# from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from models.models import KnowledgeFile
# from sqlalchemy.orm import sessionmaker
# from db.database import engine
# import numpy as np

# load_dotenv()

# # Set up database session
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Helper to build agent instructions
# def build_instructions(context: str) -> str:
#     print(f"Building instructions with context length: {len(context)} characters")
#     return (
#         "You are a friendly AI assistant. Only respond using the information provided below.\n"
#         "If a user says 'hello', 'hey', or greets you casually, reply warmly.\n"
#         "If a user asks something that's not in the context, respond: 'I don’t have information on that.'\n\n"
#         f"Context:\n{context}"
#     )


# # Custom Agent
# class Assistant(Agent):
#     def __init__(self, context: str) -> None:
#         super().__init__(instructions=build_instructions(context))

# # Function to simulate search in the KB for RAG (simple example)
# def retrieve_relevant_context(query: str, kb_files: list, top_k=3) -> str:
#     # This could be a simple keyword search, or you could use embeddings for more advanced retrieval
#     scores = []
#     for file in kb_files:
#         # Simulate scoring based on query matching (you can improve this with actual NLP models)
#         score = np.random.random()  # Placeholder for actual relevance scoring
#         scores.append((score, file.extract_data.strip()))

#     # Sort by score and return top_k relevant context chunks
#     sorted_files = sorted(scores, reverse=True, key=lambda x: x[0])
#     relevant_contexts = [text for _, text in sorted_files[:top_k]]
#     return "\n\n".join(relevant_contexts)

# # Entry point function for LiveKit worker
# async def entrypoint(ctx: JobContext):
#     kb_id = "1ddf4b7d5b024c72"
#     if not kb_id:
#         raise ValueError("Missing `kb_id` in job metadata")

#     print(f"🔍 Fetching context for KB ID: {kb_id}")
    
#     # Retrieve KB data from database
#     session = SessionLocal()
#     try:
#         kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
#     finally:
#         session.close()

#     # Use RAG approach to fetch the relevant context based on user query
#     user_query = "How does the AI assistant work?"  # Example query
#     relevant_context = retrieve_relevant_context(user_query, kb_files)

#     # Create agent with relevant context
#     agent = Assistant(context=relevant_context)

#     # Setup session components
#     session_obj = AgentSession(
#         stt=deepgram.STT(model="nova-3", language="multi"),
#         llm=openai.LLM(model="gpt-4o-mini"),
#         tts=cartesia.TTS(),
#         vad=silero.VAD.load(),
#         turn_detection=MultilingualModel(),
#     )

#     print("✅ Starting agent session in room:", ctx.room)

#     await session_obj.start(
#         room=ctx.room,
#         agent=agent,
#         room_input_options=RoomInputOptions(
#             noise_cancellation=noise_cancellation.BVC(),
#         ),
#     )

#     await ctx.connect()
#     # Generate a reply based on the user input
#     await session_obj.generate_reply(user_input=user_query)

# # Entrypoint for CLI
# if __name__ == "__main__":
#     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

# There is no official package named 'livekit-server-sdk' on PyPI.
# The official Python SDK for LiveKit is 'livekit', which you can install via:
# pip install livekit

# from livekit import LiveKitServer,LiveKitAPI
# ///////////////////////////////////////////////////
# import os

# from livekit import api
# import asyncio

# async def main():
#     # Initialize LiveKit API (replace with your server URL and API keys)
#     lkapi = api.LiveKitAPI(
#         "https://natty-gz614tko.livekit.cloud",
#         api_key="APIU7upDtrpdrEK",
#         api_secret="LWfMIqkaYAf7MgfqHnmCGkdt2jjPWeTTGxFHmvJJJHND"
#     )

#     # Create a room
#     room_info = await lkapi.room.create_room(
#         api.CreateRoomRequest(name="my-room"),
#     )
#     print("Room created:", room_info)

#     # Generate an access token for a participant
#     token = (
#         api.AccessToken()
#         .with_identity("python-bot")
#         .with_name("Python Bot")
#         .with_grants(api.VideoGrants(room_join=True, room="my-room"))
#         .to_jwt()
#     )
#     print("Participant token:", token)

#     # Clean up
#     await lkapi.aclose()

# asyncio.run(main())
# ///////////////////////////////////////////////////////////
# server.py
from livekit.api import LiveKitAPI
from livekit.api import CreateRoomRequest, ListRoomsRequest
import asyncio

from dotenv import load_dotenv
import os

load_dotenv()

print("✅ URL:", os.getenv("LIVEKIT_URL"))
print("✅ KEY:", os.getenv("LIVEKIT_API_KEY"))
print("✅ SECRET:", os.getenv("LIVEKIT_API_SECRET"))


# from livekit.api import LiveKitAPI, CreateRoomRequest, ListRoomsRequest, RoomParticipantIdentity, ListParticipantsRequest
# import asyncio

# async def main():
#     try:
#         async with LiveKitAPI() as lkapi:
        
#             room = await lkapi.room.create_room(CreateRoomRequest(
#                 name="myroom",
#                 empty_timeout=600,
#                 max_participants=10
#             ))
#             print("Room created:", room)
#     except Exception as e:
#         print("❌ Exception occurred:", e)

# if __name__ == "__main__":
#     asyncio.run(main())


# create_room.py
import os
import time
import asyncio
import jwt
from aiohttp import ClientSession
from dotenv import load_dotenv

load_dotenv()

# ✅ Load credentials from .env
API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

# ✅ Generate admin token with roomCreate permission
def generate_admin_token():
    now = int(time.time()) - 10  # small clock drift buffer
    payload = {
        "iss": API_KEY,
        "aud": "livekit",
        "iat": now,
        "exp": now + 3600,
        "video": {
            "roomCreate": True
        }
    }
    return jwt.encode(payload, API_SECRET, algorithm="HS256")

# ✅ Create room using raw HTTP request
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
