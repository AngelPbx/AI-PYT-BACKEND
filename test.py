from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from models.models import KnowledgeFile
from sqlalchemy.orm import sessionmaker
from db.database import engine

load_dotenv()

# Set up database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Helper to build agent instructions
def build_instructions(context: str) -> str:
    return (
        "You are a friendly AI assistant. Only respond using the information provided below.\n"
        "If a user says 'hello', 'hey', or greets you casually, reply warmly.\n"
        "If a user asks something that's not in the context, respond: 'I don’t have information on that.'\n\n"
        f"Context:\n{context}"
    )

# Custom Agent
class Assistant(Agent):
    def __init__(self, context: str) -> None:
        super().__init__(instructions=build_instructions(context))

# Entry point function for LiveKit worker
async def entrypoint(ctx: JobContext):
    kb_id = "1ddf4b7d5b024c72"
    if not kb_id:
        raise ValueError("Missing `kb_id` in job metadata")

    print(f"🔍 Fetching context for KB ID: {kb_id}")
    
    # Retrieve KB data from database
    session = SessionLocal()
    try:
        kb_files = session.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).all()
        context_chunks = list({f.extract_data.strip() for f in kb_files if f.extract_data})
        for idx, chunk in enumerate(context_chunks):
            print(f"--- File {idx + 1} ---\n{chunk[:500]}\n")
        combined_context = "\n\n".join(context_chunks)
    finally:
        session.close()

    # Create agent with combined context
    agent = Assistant(context=combined_context)

    # Setup session components
    session_obj = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    print("✅ Starting agent session in room:", ctx.room)
    # help(AgentSession.generate_reply)  # This should provide the function signature.


    await session_obj.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    await session_obj.generate_reply(user_input="Hello! I'm your AI assistant. How can I help you today?")

# Entrypoint for CLI
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
