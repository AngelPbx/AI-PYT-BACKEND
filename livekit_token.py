from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.agents import stt, llm
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, noise_cancellation, deepgram, silero
from sqlalchemy.orm import sessionmaker
from db.database import engine
from models.models import KnowledgeFile
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import numpy as np
from openai import OpenAI
from typing import List, AsyncIterable, Optional
import os

# Load environment variables
load_dotenv()

# Database session setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# OpenAI client
client = OpenAI()

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
    print(f"🧱 Building instructions with context length: {len(context)} characters")
    return (
        "You are a friendly AI assistant. Only respond using the information provided below.\n"
        "If a user says 'hello', 'hey', or greets you casually, reply warmly.\n"
        "If a user asks something that's not in the context, respond: 'I don’t have information on that.'\n\n"
        f"Context:\n{context}"
    )

class Assistant(Agent):
    def __init__(self, context: str, kb_files: List[KnowledgeFile]) -> None:
        super().__init__(instructions=build_instructions(context))
        self.kb_files = kb_files

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        print("\n🎤 Custom processing audio input for STT...")
        # Return the async generator from the base class directly, no await
        return Agent.default.stt_node(self, audio, model_settings)
    # async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings) -> Optional[AsyncIterable[stt.SpeechEvent]]:
    #     print("\n🎤 Custom processing audio input for STT...")
    #     base_events = Agent.default.stt_node(self, audio, model_settings)

    #     async def event_wrapper():
    #         async for event in base_events:
    #             if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
    #                 transcript = event.alternatives[0].text.lower()
    #                 if "transfer my call" in transcript:
    #                     print("⚠️ Keyword 'transfer my call' detected in STT!")
    #                     # Try to get the agent's session (not guaranteed to work!)
    #                     if hasattr(self, 'session'):
    #                         await self.session.generate_reply(
    #                             instructions="Transferring your call to a human agent now..."
    #                         )
    #             yield event

    #     return event_wrapper()



    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        print("🔄 User turn completed. Processing message...")
        user_text = new_message.text_content().strip().lower()
        print(f"🗣️ User said: {user_text}")

        if "transfer my call" in user_text:
            await self.handle_transfer_request()
            return

        rag_context = retrieve_relevant_context(user_text, self.kb_files)
        turn_ctx.add_message(role="assistant", content=f"(Contextual Info)\n{rag_context}")
        await self.update_chat_ctx(turn_ctx)

    async def handle_transfer_request(self):
        print("📞 Transfer requested. Simulating call transfer to a human agent...")
        await self.session.generate_reply(
            instructions="I am transferring your call to a human agent now. Please hold..."
        )
        # Here you could add logic to actually transfer the call, if your system supports it

async def entrypoint(ctx: JobContext):
    room = os.getenv("ROOM", "myroom")
    identity = os.getenv("IDENTITY", "agent-bot")
    print(f"🎤 Agent joining room: {room}, as identity: {identity}")
    
    kb_id = "ff60517228a84e22"
    print(f"📅 Fetching knowledge files for KB ID: {kb_id}")

    session = SessionLocal()
    try:
        kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
        print(f"📚 Found {len(kb_files)} files in the knowledge base.")
    finally:
        session.close()

    user_query = "How does the AI assistant work?"
    relevant_context = retrieve_relevant_context(user_query, kb_files)

    agent = Assistant(context=relevant_context, kb_files=kb_files)

    # Use an STT node object, not a function
    session_obj = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="coral"),
        stt=deepgram.STT(),  # or another STT node object
        turn_detection=MultilingualModel(), # or EnglishModel()
        vad=silero.VAD.load(),
    )

    print("✅ Starting agent session in room:", ctx.room)

    await session_obj.start(
        room=room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    await ctx.connect()
    


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
