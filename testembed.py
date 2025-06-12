from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from sqlalchemy.orm import sessionmaker
from db.database import engine
from models.models import KnowledgeFile

import numpy as np
from openai import OpenAI
from typing import List
import os

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

# Custom Assistant
class Assistant(Agent):
    def __init__(self, context: str) -> None:
        super().__init__(instructions=build_instructions(context))

# Entry point for LiveKit Worker
async def entrypoint(ctx: JobContext):
    kb_id = "ff60517228a84e22"  # You can later pass this from ctx.metadata
    print(f"🔍 Fetching context for KB ID: {kb_id}")
    
    session = SessionLocal()
    try:
        kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
    finally:
        session.close()

    user_query = "How does the AI assistant work?"  # This will be replaced by real user input dynamically
    relevant_context = retrieve_relevant_context(user_query, kb_files)

    agent = Assistant(context=relevant_context)

    session_obj = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
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

    await session_obj.generate_reply(user_input="Hello! I'm your AI assistant. How can I help you today?")

# 🧪 CLI Worker launch
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

