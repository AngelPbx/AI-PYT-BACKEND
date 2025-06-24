import logging
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from openai import OpenAI

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero
from models.models import KnowledgeFile
from db.database import engine

# --- Setup ---
load_dotenv()
logger = logging.getLogger("kb-agent")
logger.setLevel(logging.INFO)
client = OpenAI()
SessionLocal = sessionmaker(bind=engine)

# --- Embedding Utils ---
def get_embedding(text: str) -> list[float]:
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

def retrieve_kb_context(query: str, kb_id: str, top_k: int = 3) -> str:
    query_embedding = get_embedding(query)
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

    top_chunks = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return "\n\n".join(chunk for _, chunk in top_chunks)

# --- Shared State ---
@dataclass
class UserData:
    kb_id: str
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        return f"Agent grounded on KB ID: {self.kb_id}"

RunContext_T = RunContext[UserData]

# --- Base Agent ---
class BaseAgent(Agent):
    async def on_enter(self) -> None:
        logger.info("Assistant has entered the session.")
        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        kb_context = retrieve_kb_context("introduction", kb_id=userdata.kb_id)
        if not kb_context:
            kb_context = "No relevant knowledge base data found."

        # Inject strict KB-only prompt
        chat_ctx.add_message(
            role="system",
            content=(
                f"You are a helpful assistant. ONLY answer using the following knowledge base:\n\n"
                f"{kb_context}\n\n"
                "If the answer is not in the knowledge base, respond with: 'I'm sorry, I don't know that.'"
            )
        )
        await self.update_chat_ctx(chat_ctx)

# --- Knowledge Base Agent ---
class KBAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="Answer only based on the provided KB data. Be concise and helpful.",
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.0,
            ),
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )

    async def on_enter(self):
        await super().on_enter()
        await self.session.say("Hi, thanks for calling Angel PBX. How can I help you?")

    @function_tool
    async def transfer_to_human(self):
        """Use this function to transfer the user to a human agent."""
        await self.session.say("Transferring you to a human representative. Please hold.")
        logger.info("✅ Triggered transfer request.")

    # async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
    #     user_text = new_message.text_content.strip().lower()

    #     if "transfer" in user_text or "someone else" in user_text:
    #         await self.transfer_to_human()
    #         return

    #     await self.session.say(text=user_text)

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    kb_id = "df639e6aede94487"  # ✅ Set your actual KB ID
    userdata = UserData(kb_id=kb_id, ctx=ctx)

    session = AgentSession[UserData](userdata=userdata)
    agent = KBAgent()

    await session.start(agent=agent, room="ankit")

# --- CLI Launcher ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
