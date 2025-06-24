import numpy as np
from typing import AsyncIterable
from livekit.agents import Agent, function_tool, utils
from livekit import rtc
import logging
import os
import time
import json
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
from livekit.agents.voice import ModelSettings
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
    persona: str
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        return f"KB ID: {self.kb_id}, Persona: {self.persona}"

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
        self.volume: int = 50
        chat_ctx.add_message(
            role="system",
            content=(
                f"{userdata.persona}\n\n"
                f"ONLY answer using this knowledge base:\n{kb_context}\n\n"
                "If the answer is not in the knowledge base, say: 'I'm sorry, I don't know that.'"
                f"Your starting volume level is {self.volume}"
            )
        )
        await self.update_chat_ctx(chat_ctx)

# --- Dynamic Agent ---
class KBAgent(BaseAgent):
    def __init__(self, model_llm, model_tts, voice_tts, model_stt, lang_stt):
        super().__init__(
            instructions="Answer only based on the provided KB data. Be concise and helpful.",
            stt=deepgram.STT(model=model_stt, language=lang_stt),
            llm=openai.LLM(model=model_llm, temperature=0.0),
            tts=openai.TTS(model=model_tts, voice=voice_tts),
            vad=silero.VAD.load()
        )

    async def on_enter(self):
        await super().on_enter()
        await self.session.say("Hi, thanks for calling Angel PBX. How can I help you?")
    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        return self._adjust_volume_in_stream(
            Agent.default.tts_node(self, text, model_settings)
    )

    @function_tool()
    async def set_volume(self, volume: int):
        """Set the volume of the audio output.

        Args:
            volume (int): The volume level to set. Must be between 0 and 100.
        """
        self.volume = volume

    # Audio node used by realtime models
    async def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        return self._adjust_volume_in_stream(
            Agent.default.realtime_audio_output_node(self, audio, model_settings)
        )

    async def _adjust_volume_in_stream(
        self, audio: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[rtc.AudioFrame]:
        stream: utils.audio.AudioByteStream | None = None
        async for frame in audio:
            if stream is None:
                stream = utils.audio.AudioByteStream(
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.sample_rate // 10,  # 100ms
                )
            for f in stream.push(frame.data):
                yield self._adjust_volume_in_frame(f)

        if stream is not None:
            for f in stream.flush():
                yield self._adjust_volume_in_frame(f)

    def _adjust_volume_in_frame(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        audio_float = audio_float * max(0, min(self.volume, 100)) / 100.0
        processed = (audio_float * np.iinfo(np.int16).max).astype(np.int16)

        return rtc.AudioFrame(
            data=processed.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(processed) // frame.num_channels,
        )

    @function_tool
    async def transfer_to_human(self):
        """Use this function to transfer the user to a human agent. Only when user say transfer or someone else"""
        await self.session.say("Transferring you to a human representative. Please hold.")
        logger.info("✅ Transferring...")

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    try:
        metadata = json.loads(os.getenv("ROOM_METADATA", "{}"))
    except Exception as e:
        raise RuntimeError(f"❌ Invalid ROOM_METADATA JSON in env: {e}")

    # Extract all values from the metadata
    kb_id = metadata.get("kb_id")
    model_llm = metadata.get("model_llm", "gpt-4o-mini")
    model_tts = metadata.get("model_tts", "gpt-4o-mini-tts")
    voice_tts = metadata.get("voice_tts", "nova")
    model_stt = metadata.get("model_stt", "nova-3")
    lang_stt = metadata.get("lang_stt", "multi")
    persona = metadata.get("AGENT_PERSONA", "You are a helpful AI assistant.")

    if not kb_id:
        raise RuntimeError("❌ kb_id is required in ROOM_METADATA environment variable.")

    logger.info("✅ Loaded from env: kb_id=%s, llm=%s, tts=%s, voice=%s, stt=%s, lang=%s",
                kb_id, model_llm, model_tts, voice_tts, model_stt, lang_stt)


    userdata = UserData(kb_id=kb_id, persona=persona, ctx=ctx)
    session = AgentSession[UserData](userdata=userdata)

    agent = KBAgent(model_llm, model_tts, voice_tts, model_stt, lang_stt)
    await session.start(agent=agent, room=ctx.room.name)

# --- CLI Launcher ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


    