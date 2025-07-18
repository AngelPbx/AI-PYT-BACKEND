import logging, os, re, json, time, httpx
import numpy as np
from typing import AsyncIterable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from livekit import rtc, api
from livekit.agents import (
    JobContext, WorkerOptions, cli, function_tool, get_job_context,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
    UserInputTranscribedEvent, RunContext, AgentSession, Agent, RoomInputOptions, ModelSettings, function_tool, utils
)
from livekit.plugins import deepgram, openai, silero, noise_cancellation,elevenlabs
from models.models import KnowledgeFile, PBXLLM, pbx_ai_agent, WebCall
from db.database import engine

# agent knowledgebase done
# voice control done
# custom function call for sample done
# end call function
# pronunciation part done
# Adding background audio sample done but can't check in local
# transcript with data stored in chat table in db done 

# --- Setup ---
load_dotenv()
logger = logging.getLogger("kb-agent")
logger.setLevel(logging.INFO)
client = OpenAI()
SessionLocal = sessionmaker(bind=engine)


def get_agent_and_llm(agent_id: str):
    """Fetch Agent and LLM directly from DB"""
    db = SessionLocal()
    try:
        # Fetch agent
        agent = db.query(pbx_ai_agent).filter(pbx_ai_agent.id == agent_id).first()
        if not agent:
            raise ValueError(f"No Agent found with ID: {agent_id}")

        # Fetch LLM using llm_id from agent.response_engine
        llm_id = agent.response_engine.get("llm_id")
        llm = db.query(PBXLLM).filter(PBXLLM.id == llm_id).first()
        if not llm:
            raise ValueError(f"No PBXLLM found with ID: {llm_id}")

        return agent, llm
    finally:
        db.close()

def build_stt(s2s_model: str = None, language: str = "en"):
    """Dynamically build STT engine based on model"""
    # Fallback to whisper if s2s_model is None
    if not s2s_model:
        return openai.STT(language=language)

    s2s_model = s2s_model.lower()

    if "whisper" in s2s_model:
        return openai.STT(language=language)
    elif "deepgram" in s2s_model:
        return deepgram.STT(model=s2s_model)
    else:
        # Default to Whisper as fallback
        return openai.STT(language=language)

def build_tts(agent):
    """Dynamically build TTS   openai-tts is in agent model""" 
    if agent.voice_model == "openai-tts":
        return openai.TTS(
            voice=agent.voice_id,
            speed=agent.voice_speed or 1.0,
            temperature=agent.voice_temperature or 0.7,
        )
    elif agent.voice_model == "elevenlabs":
        return elevenlabs.TTS(
            voice_id=agent.voice_id,
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )
    else:
        return elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )
        


def build_llm(llm):
    """Dynamically build LLM"""
    return openai.LLM(
        model=llm.model,
        temperature=llm.model_temperature or 0.7,
    )

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

async def hangup_call():
    ctx = get_job_context()
    if ctx is None:
        # Not running in a job context
        return
    
    await ctx.api.room.delete_room(
        api.DeleteRoomRequest(
            room=ctx.room.name,
        )
    )
# --- Shared State ---
@dataclass
class UserData:
    kb_id: str
    persona: str
    begin_message: Optional[str] = None
    ctx: Optional[JobContext] = None
    start_timestamp: Optional[int] = None
    def summarize(self) -> str:
        return f"KB ID: {self.kb_id}, Persona: {self.persona}"

# --- Agent ---
class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="Answer only based on the provided KB data. Be concise and helpful.")
        self.volume: int = 50
        self.transcript_log: list[str] = []

    async def on_enter(self):
        userdata: UserData = self.session.userdata
        userdata.start_timestamp = int(time.time() * 1000)
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
        await self.session.say(f"{userdata.begin_message}")

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
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await hangup_call()
   
    @function_tool
    async def transfer_to_human(self):       
        await self.session.say("Transferring you to a human representative. Please hold.")
       

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    metadata = json.loads(ctx.job.metadata)
    agent_id = metadata.get("agent_id")
   
    agent, llm = get_agent_and_llm(agent_id)

    stt = build_stt(llm.s2s_model, language=agent.language or "en")
    tts = build_tts(agent)
    llm_plugin = build_llm(llm)
    begin_message=llm.begin_message
    persona = llm.general_prompt
    kb_ids = llm.knowledge_base_ids or []
    kb_id = kb_ids[0] if kb_ids else None

    userdata = UserData(kb_id=kb_id, persona=persona,begin_message=begin_message ,ctx=ctx)


    session = AgentSession(
        userdata=userdata,
        stt=stt,
        tts=tts,
        llm=llm_plugin,
        vad=silero.VAD.load(),
    )
    
    agent = Assistant()

    @session.on("user_input_transcribed")
    def on_transcribed(event: UserInputTranscribedEvent):
        if event.is_final:
            agent.transcript_log.append(f"User: {event.transcript}")
           

    async def analyze_chat(session_history: dict) -> dict:
        """
        Analyze the chat transcript using OpenAI to generate:
        - Summary
        - Sentiment
        - Chat success
        """
        messages = session_history.get("items", [])

        # Build formatted chat string
        full_chat = "\n".join(
            f"{item['role'].capitalize()}: {' '.join(item['content'])}"
            for item in messages
            if item.get("type") == "message" and item.get("content")
        )

        prompt = (
            "You are a call center assistant analytics AI.\n"
            "Analyze the following conversation and return *only* a JSON object with these keys:\n"
            "  - chat_summary (1-2 sentence summary of the conversation)\n"
            "  - user_sentiment (Positive, Neutral, or Negative based on user's tone)\n"
            "  - chat_successful (true if user query was addressed, false otherwise)\n\n"
            f"Transcript:\n{full_chat}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                
            )
        
            content = response.choices[0].message.content.strip()
           
            match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
            if match:
                json_text = match.group(1)
            else:
               
                json_text = content

            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as je:
                logging.error("Failed to parse JSON from analysis response: %s", je)
                raise

            return {
                "chat_summary":     result.get("chat_summary",     "Summary not available."),
                "user_sentiment":   result.get("user_sentiment",   "Neutral"),
                "chat_successful":  bool(result.get("chat_successful", False)),
            }

        except Exception as e:
            logging.exception("❌ Chat analysis failed")
            return {
                "chat_summary":   "Analysis failed.",
                "user_sentiment": "Unknown",
                "chat_successful": False
            }
    from datetime import datetime
    async def write_transcript():
        try:
            
            db = SessionLocal()
            # Get the call_id from the room name or metadata
            call_id = ctx.room.name  # Assuming room.name == call_id

            # Fetch WebCall record
            web_call = db.query(WebCall).filter(WebCall.call_id == call_id).first()
            if not web_call:
                return

            # Save transcript data
            transcript_data = session.history.to_dict()
            web_call.transcript_object = transcript_data
            web_call.transcript = "\n".join([
                f"{'Agent' if msg['role'] == 'assistant' else 'User'}: {msg['content'][0]}"
                for msg in transcript_data.get('items', [])
                if msg['type'] == 'message' and 'content' in msg and msg['content']
            ])
            web_call.updated_at = int(time.time() * 1000)

            db.commit()

        except Exception as e:
            print(f"Error saving transcript to DB: {e}")
        finally:
            db.close()

    ctx.add_shutdown_callback(write_transcript)
    
   

    # Optional: Background audio
    # background_audio = BackgroundAudioPlayer(
    #     ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
    #     thinking_sound=[
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
    #     ],
    # )
    # participant = await ctx.wait_for_participant()
    # if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
    #     call_id = participant.attributes.get("sip.callID")
    #     phone = participant.attributes.get("sip.phoneNumber", "unknown")
    #     print(f"📞 SIP call from {phone}, Call ID: {call_id}")

    await session.start(agent=agent,room=ctx.room,
                         room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ))
    
    # await background_audio.start(agent_session=session, room=ctx.room)
    await ctx.connect()

# --- CLI ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,agent_name=os.getenv("ROOM")))