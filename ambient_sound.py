import os
import re
import json
import time
import logging
import numpy as np
from typing import AsyncIterable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
import numpy as np
from livekit.agents import Agent, function_tool, utils
from openai import OpenAI
from livekit import rtc
from livekit.agents import (
    JobContext, WorkerOptions, cli, function_tool,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
    UserInputTranscribedEvent, RunContext
)
from livekit.agents import AgentSession, Agent, RoomInputOptions, ModelSettings

from livekit.plugins import deepgram, openai, silero, noise_cancellation
from models.models import KnowledgeFile
from db.database import engine

# agent knowledgebase done
# voice volume control done
# custom function call for sample done
# end call function
# pronunciation part done
# Adding background audio sample done but can't check in local
# transcript sample done

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
        model="text-embedding-3-small", input=[text]
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

# --- Agent ---
class KBAgent(Agent):
    def __init__(self):
        super().__init__(instructions="Answer only based on the provided KB data. Be concise and helpful.")
        self.volume: int = 50

    async def on_enter(self):
        logger.info("✅ Agent entered session.")
        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        kb_context = retrieve_kb_context("introduction", kb_id=userdata.kb_id)
        if not kb_context:
            kb_context = "No relevant knowledge base data found."

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
        await self.session.say("Hi, thanks for calling Angel PBX. How can I help you?")

    async def tts_node(
    self, 
    text: AsyncIterable[str], 
    model_settings: ModelSettings
) -> AsyncIterable[rtc.AudioFrame]:
        pronunciations = {
            "API": "A P I", "book": "Booooooks", "REST": "rest", "SQL": "sequel",
            "kubectl": "kube control", "AWS": "A W S", "UI": "U I", "URL": "U R L",
            "npm": "N P M", "LiveKit": "Live Kit", "async": "a sink", "nginx": "engine x",
        }

        async def adjust_pronunciation(input_text: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in input_text:
                for term, pronunciation in pronunciations.items():
                    chunk = re.sub(rf'\b{term}\b', pronunciation, chunk, flags=re.IGNORECASE)
                yield chunk

        # 👇 Chain both: apply pronunciation, then volume control
        return self._adjust_volume_in_stream(
            Agent.default.tts_node(self, adjust_pronunciation(text), model_settings)
    )


    # async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[rtc.AudioFrame]:
    #     pronunciations = {
    #         "API": "A P I", "book": "Booooooks", "REST": "rest", "SQL": "sequel",
    #         "kubectl": "kube control", "AWS": "A W S", "UI": "U I", "URL": "U R L",
    #         "npm": "N P M", "LiveKit": "Live Kit", "async": "a sink", "nginx": "engine x",
    #     }

    #     async def adjust_pronunciation(input_text: AsyncIterable[str]) -> AsyncIterable[str]:
    #         async for chunk in input_text:
    #             for term, pronunciation in pronunciations.items():
    #                 chunk = re.sub(rf'\b{term}\b', pronunciation, chunk, flags=re.IGNORECASE)
    #             yield chunk

    #     async for frame in Agent.default.tts_node(self, adjust_pronunciation(text), model_settings):
    #         yield frame

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
    async def end_call(self, ctx: RunContext[UserData]):
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

    @function_tool
    async def transfer_to_human(self):
        await self.session.say("Transferring you to a human representative. Please hold.")
        logger.info("✅ Transferring...")

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    try:
        metadata = json.loads(os.getenv("ROOM_METADATA", "{}"))
    except Exception as e:
        raise RuntimeError(f"❌ Invalid ROOM_METADATA JSON in env: {e}")

    kb_id = metadata.get("kb_id")
    if not kb_id:
        raise RuntimeError("❌ kb_id is required in ROOM_METADATA environment variable.")
    persona = metadata.get("AGENT_PERSONA", "You are a helpful AI assistant.")

    userdata = UserData(kb_id=kb_id, persona=persona, ctx=ctx)

    session = AgentSession(
        userdata=userdata,
        llm=openai.LLM(model=metadata.get("model_llm", "gpt-4o-mini")),
        tts=openai.TTS(
            model=metadata.get("model_tts", "gpt-4o-mini-tts"),
            voice=metadata.get("voice_tts", "nova")
        ),
        stt=deepgram.STT(
            model=metadata.get("model_stt", "nova-3"),
            language=metadata.get("lang_stt", "multi")
        ),
        vad=silero.VAD.load(),
       
    )

    # Optional: Handle events like transcription
    @session.on("user_input_transcribed")
    def on_transcribed(event: UserInputTranscribedEvent):
        print(f"[📝 Transcript] {event.transcript} (final={event.is_final})")

    # Optional: Background audio
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    await session.start(agent=KBAgent(),room=ctx.room,
                         room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ))
    await background_audio.start(agent_session=session, room=ctx.room)

# --- CLI ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
