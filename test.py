#save test py for agent ## natty
import asyncio
import os,json
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.plugins import openai, noise_cancellation
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select,create_engine
from models.models import KnowledgeFile
import logging
from openai import OpenAI
import numpy as np
from dataclasses import dataclass
from typing import AsyncIterable, Optional
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    elevenlabs
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    response = client.embeddings.create(
        model="text-embedding-3-small", input=[text]
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    if not a or not b:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def _perform_search(query: str, kb_id: str = None, top_k: int = 3) -> str:
    logger.info(f"✅Performing vector-based KB search for query: '{query}' | kb_id: {kb_id}")
    
    query_embedding = get_embedding(query)  # Make sure this is sync or wrap in asyncio.to_thread()

    session = SessionLocal()
    results = []

    try:
         kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
         for file in kb_files:

            if not file.extract_data:
                continue
            emb = file.embedding
            if isinstance(emb, str):
                emb = np.array(eval(emb))                
            score = cosine_similarity(query_embedding, emb)
            results.append((score, file.extract_data.strip()))
    finally:
        session.close()

    if not results:
        logger.info("⚠️ No matching embeddings found.")
        return "No relevant information found in the knowledge base."

    top_chunks = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    logger.info(f"✅Top {len(top_chunks)} results selected.")
    return "\n\n".join(chunk for _, chunk in top_chunks)

# Tool with status update fallback
@function_tool()
async def search_knowledge_base(context: RunContext, query: str) -> str:
    logger.info(f"✅ Tool triggered: search_knowledge_base | query='{query}'")
    userdata = context.session.userdata  # Get UserData from session
    kb_id = userdata.kb_id  # Extract kb_id

    logger.info(f"✅✅✅✅ Using kb_id: {kb_id}")


    if not kb_id:
        logger.warning("❌ No kb_id found in session metadata.")
        return "Knowledge base ID is missing. Cannot search."

    async def _speak_status_update(delay: float = 0.5):
        await asyncio.sleep(delay)
        logger.info("✅Status update: search taking longer than expected.")
        await context.session.generate_reply(
            instructions=f"""You are searching the knowledge base for "{query}" but it's taking a little while. Briefly update the user about the progress."""
        )

    status_update_task = asyncio.create_task(_speak_status_update())

    try:
        result = await asyncio.to_thread(_perform_search, query, kb_id)
        if not status_update_task.done():
            status_update_task.cancel()
            try:
                await status_update_task
            except asyncio.CancelledError:
                logger.info("✅ Status update task cancelled cleanly.")
        # status_update_task.cancel()
        return result or "⚠️ No relevant information found in the knowledge base."
    except asyncio.CancelledError:
        logger.info("⚠️ Status update task cancelled due to fast response.")
        return "⚠️ Search was interrupted before completion."  # ✅ Added this
    except Exception as e:
        logger.error(f"⚠️ Error during KB search: {e}")
        return f"⚠️ An error occurred while searching: {e}"


@dataclass
class UserData:
    kb_id: str
    persona: str
    ctx: Optional[agents.JobContext] = None



# Define your agent with tools
class Assistant(Agent):
    def __init__(self, persona: str = None) -> None:
        super().__init__(
             instructions=f"""{persona}.You MUST answer user questions by calling the tool 'search_knowledge_base'.
    DO NOT answer from your own knowledge. If the tool does not return a result, tell the user you could not find relevant information.""",
            
            tools=[search_knowledge_base]
        )


async def entrypoint(ctx: agents.JobContext):
    metadata = json.loads(ctx.job.metadata)
    kb_id = metadata.get("kb_id")
    persona = metadata.get("persona")
    logger.info(f"✅ Starting agent | kb_id={kb_id}, persona={persona}")
    userdata = UserData(kb_id=kb_id, persona=persona, ctx=ctx)

# llm congiguration
    session = AgentSession(
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
        vad=silero.VAD.load(),
   
    )
# ////////////////////////////


    await session.start(
        room=ctx.room,
        agent=Assistant(persona=persona),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
