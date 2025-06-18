from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.agents import stt
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, noise_cancellation, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from sqlalchemy.orm import sessionmaker
from db.database import engine
from models.models import KnowledgeFile

import numpy as np
from openai import OpenAI
from typing import List, AsyncIterable, Optional
import os, json

# Load environment variables
load_dotenv()

# Setup DB session
SessionLocal = sessionmaker(bind=engine)

# OpenAI client
client = OpenAI()


# ---------- Embedding Logic ----------
def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    if not a or not b:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_context(query: str, kb_files: List[KnowledgeFile], top_k=3) -> str:
    print(f"🔍 Retrieving relevant context for query: '{query}'")
    query_embedding = get_embedding(query)
    scored_chunks = []

    for file in kb_files:
        if not file.extract_data:
            continue
        try:
            file_embedding = file.embedding
            if isinstance(file_embedding, str):
                file_embedding = np.array(eval(file_embedding))
            score = cosine_similarity(query_embedding, file_embedding)
            scored_chunks.append((score, file.extract_data.strip()))
        except Exception as e:
            print(f"⚠️ Embedding failed for file {file.filename}: {e}")

    top_chunks = sorted(scored_chunks, reverse=True, key=lambda x: x[0])[:top_k]
    return "\n\n".join(chunk for _, chunk in top_chunks)


def build_instructions(context: str) -> str:
    print(f"🧱 Building instructions with context length: {len(context)}")
    return (
        "You are a friendly AI assistant. Only respond using the information below.\n"
        "If the question is unrelated, say: 'I don’t have information on that.'\n\n"
        f"Context:\n{context}"
    )


# ---------- Assistant Agent ----------
class Assistant(Agent):
    def __init__(self, kb_id: str) -> None:
        super().__init__(instructions="")  # Leave instructions blank
        self.kb_id = kb_id

    def use_llm(self) -> bool:
        return True

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings):
        print("🎤 Custom STT processing")
        return super().stt_node(audio, model_settings)

    async def on_enter(self):
        print("👋 Sending welcome greeting...")
        await self.session.generate_reply(
            user_input="Say hello to the user",
            instructions="You are a helpful agent. Greet the user."
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        user_text = new_message.text_content.strip()
        print(f"\n🗣️ User: {user_text}")

        if "transfer my call" in user_text.lower():
            await self.handle_transfer_request()
            turn_ctx.responded = True
            return

        session = SessionLocal()
        try:
            kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == self.kb_id).all()
        finally:
            session.close()

        context = retrieve_relevant_context(user_text, kb_files)
        instructions = build_instructions(context)

        print("💬 Generating LLM response...")
       
        turn_ctx.responded = True

    async def handle_transfer_request(self):
        await self.session.generate_reply(
            user_input="transfer my call",
            instructions="Tell the user you're transferring them to a human."
        )


# ---------- Entrypoint ----------1821485571600
async def entrypoint(ctx: JobContext):
    room = os.getenv("ROOM", "myroom")
    identity = os.getenv("IDENTITY", "agent-bot")
    print(f"🔗 Connecting to room: {room} as {identity}")

    metadata = ctx.room.metadata
    if not isinstance(metadata, str) or not metadata.strip():
        metadata = os.getenv("ROOM_METADATA", "{}")
   
    print(f"📜 Room metadata: {metadata}")

    try:
        if isinstance(metadata, str):
            meta = json.loads(metadata)
        else:
            meta = json.loads(str(metadata))
        kb_id = meta.get("kb_id", "Not provided")
        print(f"✅ kb_id: {kb_id}")
    except Exception as e:
        print(f"⚠️ Failed to parse metadata: {e}")
        kb_id = None

    if not kb_id or kb_id == "Not provided":
        print("❌ kb_id not provided in room metadata.")
        return
        
    # kb_id = "df639e6aede94487"  # Your knowledge base ID

    print(f"🎤 Joining room {room} as {identity}")
    agent = Assistant(kb_id=kb_id)

    session_obj = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="nova"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session_obj.start(
        room=room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        )
    )
    await ctx.connect()


# ---------- CLI ----------
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
