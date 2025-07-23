import logging, os, re, json, time, asyncio
import numpy as np
from typing import AsyncIterable, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from livekit import rtc, api
from livekit.agents import metrics, MetricsCollectedEvent
from livekit.agents import (
    JobContext, WorkerOptions, cli, function_tool, get_job_context,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
    UserInputTranscribedEvent, RunContext, AgentSession, Agent, RoomInputOptions, ModelSettings, function_tool, utils
)
from livekit.plugins import deepgram, openai, silero, noise_cancellation,elevenlabs
from models.models import KnowledgeFile, PBXLLM, pbx_ai_agent, WebCall
from db.database import engine
from livekit.agents import ConversationItemAddedEvent
from livekit.agents.llm import AudioContent
from livekit.agents.voice.events import CloseEvent, ErrorEvent
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

        # Prepare pronunciation dictionary
        pronunciations = {}
        for item in agent.pronunciation_dictionary or []:
            word = item.get("word")
            phoneme = item.get("phoneme")
            if word and phoneme:
                pronunciations[word] = phoneme

        return agent, llm, pronunciations
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
        return deepgram.STT(
        model="general",  # ✅ fallback to general model
        language="en-US"
    )
        # return deepgram.STT(model=s2s_model)
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
            api_key=os.getenv("ELEVENLABS_API_KEY"),
           
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


def retrieve_kb_context(query: str, kb_ids: List[str], top_k: int = 3) -> str:
    query_embedding = get_embedding(query)
    session = SessionLocal()
    results = []
    try:
        kb_files = (
            session.query(KnowledgeFile)
            .filter(KnowledgeFile.kb_id.in_(kb_ids))
            .all()
        )

      
        for file in kb_files:
            if not file.extract_data:
                continue
            emb = file.embedding
            if isinstance(emb, str):
                emb = np.array(eval(emb))                
            score = cosine_similarity(query_embedding, emb)
            results.append({
                "score": score,
                "content": file.extract_data.strip(),
                "file_path": file.file_path,
            })
           
    finally:
        session.close()

    top_chunks = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Create combined context
    combined_context = "\n\n".join(chunk["content"] for chunk in top_chunks)

    return combined_context, top_chunks
   
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
    kb_ids: List[str]
    persona: str
    begin_message: Optional[str] = None
    ctx: Optional[JobContext] = None
    start_timestamp: Optional[int] = None
    retrieved_file_paths: List[dict] = None
    pronunciations: dict = field(default_factory=dict)
    llm_token_usage: dict = field(default_factory=lambda: {
        "values": [],
        "average": 0,
        "num_requests": 0
    })  
    def summarize(self) -> str:
        return f"KB ID: {self.kb_ids}, Persona: {self.persona}"

# --- Agent ---
class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="Answer only based on the provided KB data. Be concise and helpful.",
                         tools=[])
        self.volume: int = 50
        self.transcript_log: list[str] = []

    async def on_enter(self):
        userdata: UserData = self.session.userdata
        userdata.start_timestamp = int(time.time() * 1000)
        chat_ctx = self.chat_ctx.copy()

        # kb_context = retrieve_kb_context("introduction", kb_ids=userdata.kb_ids)
        kb_context, top_chunks = retrieve_kb_context("introduction", kb_ids=userdata.kb_ids)

        userdata.retrieved_file_paths = [
            {               
                "file_path": chunk["file_path"],
            }
            for chunk in top_chunks
            if chunk.get("file_path")  # Only include if file_path is not None
        ]

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
        userdata: UserData = self.session.userdata
        pronunciations_test = {'API': 'A P I', 'book': 'Boooooks','Ankit': 'aaaaankeeeet' ,'SQL': 'sequel'}
        logging.info(f"⚠️⚠️⚠️ Pronunciations:{type(userdata.pronunciations)} {userdata.pronunciations.items()}---------")

        async def adjust_pronunciation(input_text: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in input_text:
                for term, phoneme in userdata.pronunciations.items():
                    # chunk = re.sub(rf'\b{re.escape(term)}\b',phoneme,chunk,flags=re.IGNORECASE)
                    # chunk = re.sub(rf'\b{term}\b',phoneme,chunk,flags=re.IGNORECASE)
                   # Escape regex special characters in term
                    safe_term = re.escape(term)
                    # Replace all occurrences, ignoring case 
                    chunk = re.sub(safe_term, phoneme, chunk, flags=re.IGNORECASE)
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
   
    agentdb, llm, pronunciations  = get_agent_and_llm(agent_id)

    stt = build_stt(llm.s2s_model, language=agentdb.language or "en")
    tts = build_tts(agentdb)
    llm_plugin = build_llm(llm)
    begin_message=llm.begin_message
    persona = llm.general_prompt
    kb_ids = llm.knowledge_base_ids or []
    # kb_id = kb_ids[0] if kb_ids else None

    userdata = UserData(kb_ids=kb_ids, persona=persona,begin_message=begin_message ,pronunciations =pronunciations ,ctx=ctx)


    session = AgentSession(
        userdata=userdata,
        stt=stt,
        tts=tts,
        llm=llm_plugin,
        vad=silero.VAD.load(),
    )
    
    agent = Assistant()

    # Add error and close handlers
    # @session.on("error")
    # def on_error(ev: ErrorEvent):
    #     if ev.error.recoverable:
    #         return
    #     logger.warning(f"⚠️ Error from {ev.source}: {ev.error}")

    #     # Make agent say fallback message
    #     session.say(
    #         f"I'm having trouble right now. Please try again. {ev.source}: {ev.error}",
    #         allow_interruptions=False,
    #     )



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
        full_chat = "\n".join(
            f"{'Agent' if msg['role'] == 'assistant' else 'User'}: {' '.join(msg['content'])}"
            for msg in messages
            if msg.get("type") == "message" and msg.get("content")
        )

        prompt = (
            "You are a call center assistant analytics AI.\n"
            "Analyze the following conversation and return ONLY a JSON object with these keys:\n"
            "- chat_summary (1-2 sentence summary)\n"
            "- user_sentiment (Positive, Neutral, or Negative)\n"
            "- chat_successful (true or false)\n\n"
            f"Transcript:\n{full_chat}\n\n"
            "Return the JSON object directly without markdown or explanation."
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
            )
        )
            # 👇 Track token usage
            total_tokens = response.usage.total_tokens

            # ✅ Track usage properly
            llm_tokens = userdata.llm_token_usage
            llm_tokens["values"].append(total_tokens)
            logging.info(f"LLM Tokens used:✅✅👇✅ {total_tokens}")
            llm_tokens["num_requests"] = len(llm_tokens["values"])
            llm_tokens["average"] = sum(llm_tokens["values"]) / llm_tokens["num_requests"]


            # userdata.llm_token_usage["values"].append(total_tokens)
            # userdata.llm_token_usage["num_requests"] += 1


            content = response.choices[0].message.content.strip()

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # fallback to regex parse if wrapped in ```json``` fences
                match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    raise ValueError("Response is not valid JSON")

            return {
                "chat_summary": result.get("chat_summary", "Summary not available."),
                "user_sentiment": result.get("user_sentiment", "Neutral"),
                "chat_successful": bool(result.get("chat_successful", False)),
            }

        except asyncio.TimeoutError:
            logging.error("❌ Chat analysis timed out")
        except Exception as e:
            logging.exception("❌ Chat analysis failed")

        return {
            "chat_summary": "Analysis failed.",
            "user_sentiment": "Unknown",
            "chat_successful": False
        }

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
            raw_items = session.history.to_dict().get("items", [])
            cleaned_transcript = [
                    {
                        "role": "agent" if item["role"] == "assistant" else item["role"],
                        "content": " ".join(item["content"])
                    }
                    for item in raw_items if item.get("type") == "message"
                ]

            web_call.transcript_object = cleaned_transcript

            # web_call.transcript_object = transcript_data
            web_call.transcript = "\n".join([
                f"{'Agent' if msg['role'] == 'assistant' else 'User'}: {msg['content'][0]}"
                for msg in transcript_data.get('items', [])
                if msg['type'] == 'message' and 'content' in msg and msg['content']
            ])
            web_call.updated_at = int(time.time() * 1000)
            web_call.call_status = "ended"
            values = userdata.llm_token_usage["values"]
            userdata.llm_token_usage["average"] = (
                sum(values) / len(values) if values else 0
            )

            web_call.llm_token_usage = userdata.llm_token_usage
            if userdata.retrieved_file_paths:
                file_paths = [item["file_path"] for item in userdata.retrieved_file_paths]
                web_call.knowledge_base_retrieved_contents_url = ", ".join(file_paths)
            else:
                pass

            end_ts = int(time.time() * 1000)
            if web_call.start_timestamp:
                web_call.end_timestamp = end_ts
                web_call.duration_ms = end_ts - web_call.start_timestamp
            else:
                # fallback in case start_timestamp wasn't set earlier
                web_call.start_timestamp = end_ts
                web_call.end_timestamp = end_ts
                web_call.duration_ms = 0

            analysis_result = await analyze_chat(transcript_data)
            web_call.call_analysis = {
                "in_voicemail": False,  # set to True if detected during call
                "call_summary": analysis_result.get("chat_summary"),
                "user_sentiment": analysis_result.get("user_sentiment"),
                "custom_analysis_data": {},  # optionally extend with custom metrics
                "call_successful": analysis_result.get("chat_successful"),
            }


            db.commit()

        except Exception as e:
            print(f"Error saving transcript to DB: {e}")
        finally:
            db.close()

    ctx.add_shutdown_callback(write_transcript)
# //////////////////////////////
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: 📞📞📞{summary}")

    # At shutdown, generate and log the summary from the usage collector
    ctx.add_shutdown_callback(log_usage)
# ////////////////////////////////////
    await ctx.connect()
    await session.start(agent=agent,room=ctx.room,
                         room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ))
    # An audio player with automated ambient and thinking sounds

    # 🌟 Map API values to file names or built-in clips
    AMBIENT_SOUND_MAP = {
        "forest-birds": "forestbirds.wav",
        "coffee-shop": "coffee-shop.wav",
        "convention-hall": "convention-hall.wav",
        "summer-outdoor": "summer-outdoor.wav",
        "office": BuiltinAudioClip.OFFICE_AMBIENCE,
        "mountain-outdoor": "mountain-outdoor.wav",
        "static_-noise": "static-noise.wav",
        "call-center": "call-center.wav",
        "none": None  # no ambient sound
    }
    # logger.info("🌟 Ambient sound map initialized.",agentdb.ambient_sound)

    # Get ambient sound from agent API metadata
    ambient_sound_key = agentdb.ambient_sound or "none"  # fallback to "none"
    ambient_sound_volume_key = agentdb.ambient_sound_volume or "0.8"  # default volume

    # Resolve to actual audio config
    ambient_sound_value = AMBIENT_SOUND_MAP.get(ambient_sound_key, None)
    if ambient_sound_value is None:
        logger.info("☕ No ambient sound selected.")
    else:
        logger.info(f"🎵 Ambient sound selected: {ambient_sound_key}")

    # ✅ Setup BackgroundAudioPlayer
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(ambient_sound_value, volume=ambient_sound_volume_key) if ambient_sound_value else None,
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    # ✅ Start and play ambient sound
    await background_audio.start(agent_session=session, room=ctx.room)
    if ambient_sound_value:
        await background_audio.play(ambient_sound_value, loop=True)


    # # ✅ Only one BackgroundAudioPlayer
    # background_audio = BackgroundAudioPlayer(
    #     ambient_sound=AudioConfig("forestbirds.wav", volume=0.8),
    #     thinking_sound=[
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
    #         AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
    #     ],
    # )

    # # ✅ Start background audio linked to session
    # await background_audio.start(agent_session=session, room=ctx.room)

    # # ✅ Play ambient sound (loops automatically)
    # await background_audio.play("forestbirds.wav", loop=True)

    
    
    
    

# --- CLI ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,agent_name=os.getenv("ROOM")))