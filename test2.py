from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.deepgram import STT as DeepgramSTT

from sqlalchemy.orm import sessionmaker
from db.database import engine
from models.models import KnowledgeFile

import numpy as np
from openai import OpenAI
from typing import List, Optional, Callable
import os
import asyncio

# Load environment variables
load_dotenv()

# DB session setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# OpenAI client
client = OpenAI()

# 🧠 Embedding helper
def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

# 📚 Vector similarity search using cosine similarity
def cosine_similarity(a, b):
    if not a or not b:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 🔍 RAG Retriever using query + document embedding match
def retrieve_relevant_context(query: str, kb_files: List[KnowledgeFile], top_k=3) -> str:
    query_embedding = get_embedding(query)
    scored_chunks = []

    for file in kb_files:
        if not file.extract_data:
            continue

        try:
            file_embedding = file.embedding  # Already stored in DB
            if isinstance(file_embedding, str):  # In case it's stored as JSON/text
                file_embedding = np.array(eval(file_embedding))
            score = cosine_similarity(query_embedding, file_embedding)
            scored_chunks.append((score, file.extract_data.strip()))
        except Exception as e:
            print(f"Embedding failed for file {file.filename}: {e}")

    top_chunks = sorted(scored_chunks, reverse=True, key=lambda x: x[0])[:top_k]
    return "\n\n".join(chunk for _, chunk in top_chunks)

# 💬 Instruction builder for context-limited AI responses
def build_instructions(context: str) -> str:
    print(f"Building instructions with context length: {len(context)} characters")
    return (
        "You are a friendly AI assistant. Only respond using the information provided below.\n"
        "If a user says 'hello', 'hey', or greets you casually, reply warmly.\n"
        "If a user asks something that's not in the context, respond: 'I don’t have information on that.'\n\n"
        f"Context:\n{context}"
    )

# ➕ Function to print 2+2
def print_two_plus_two():
    print("2 + 2 =", 2 + 2)

# Custom Assistant
class Assistant(Agent):
    def __init__(self, context: str) -> None:
        super().__init__(instructions=build_instructions(context))

# Custom STT Handler for listening to user speech
class CustomSTT:
    def __init__(self, stt: DeepgramSTT):
        self.stt = stt
        self.on_speech: Optional[Callable[[str], None]] = None

    async def transcribe(self, audio_stream):
        async for event in self.stt.transcribe(audio_stream):
            if event.type == "transcription" and event.alternatives:
                text = event.alternatives[0].text
                if self.on_speech:
                    self.on_speech(text)

    def set_on_speech(self, callback: Callable[[str], None]):
        self.on_speech = callback

# Simulated audio stream for testing
async def simulate_audio_stream(stt_handler: CustomSTT):
    # Simulate user saying "transfer my call"
    await asyncio.sleep(1)
    # Simulate transcription event
    if stt_handler.on_speech:
        stt_handler.on_speech("transfer my call")

# Entry point for LiveKit Worker
async def entrypoint(ctx: JobContext):
    kb_id = "ff60517228a84e22"
    print(f"🔍 Fetching context for KB ID: {kb_id}")
    
    session = SessionLocal()
    try:
        kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
    finally:
        session.close()

    user_query = "How does the AI assistant work?"  # Default, but overridden by real speech
    relevant_context = retrieve_relevant_context(user_query, kb_files)
    agent = Assistant(context=relevant_context)

    # Create custom STT with event handler
    stt = DeepgramSTT(model="nova-3", language="multi")
    custom_stt = CustomSTT(stt)

    def on_user_speech(text: str):
        print("User said:", text)
        if "transfer my call" in text.lower():
            print_two_plus_two()

    custom_stt.set_on_speech(on_user_speech)

    # In a real app, you would get the audio stream from the room/participant
    # and pass it to custom_stt.transcribe(audio_stream)
    # For testing, we use a simulated audio stream
    asyncio.create_task(simulate_audio_stream(custom_stt))

    session_obj = AgentSession(
        # stt is not used here, since we use custom_stt
        llm=openai.LLM(model="gpt-3.5-turbo"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    print("✅ Starting agent session in room:", ctx.room)

    await session_obj.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Simulate a reply
    await session_obj.generate_reply(user_input="Hello! I'm your AI assistant. How can I help you today?")

    # Keep the task running to simulate ongoing transcription
    while True:
        await asyncio.sleep(1)

# 🧪 CLI Worker launch
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
