import logging, os, re, json
import numpy as np
from typing import AsyncIterable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from livekit import rtc
from livekit.agents import (
    JobContext, WorkerOptions, cli, function_tool,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
    UserInputTranscribedEvent, RunContext, AgentSession, Agent, RoomInputOptions, ModelSettings, function_tool, utils
)
from livekit.plugins import deepgram, openai, silero, noise_cancellation
from models.models import KnowledgeFile
from db.database import engine
import httpx
import time

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
    start_timestamp: Optional[int] = None
    def summarize(self) -> str:
        return f"KB ID: {self.kb_id}, Persona: {self.persona}"

# --- Agent ---
class KBAgent(Agent):
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
        if ctx.userdata.ctx:
            ctx.userdata.ctx.shutdown()

    @function_tool
    async def transfer_to_human(self):       
        await self.session.say("Transferring you to a human representative. Please hold.")
       

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    
    raw_metadata = ctx.room.metadata
    if not isinstance(raw_metadata, str) or not raw_metadata.strip():
        raw_metadata = os.getenv("ROOM_METADATA", "{}")
   
    try:
        if isinstance(raw_metadata, str):
            metadata = json.loads(raw_metadata)
        else:
            metadata = json.loads(str(raw_metadata))
        kb_id = metadata.get("kb_id", "Not provided")
    except Exception as e:
        print(f"⚠️ Failed to parse metadata: {e}")
        kb_id = None

    if not kb_id or kb_id == "Not provided":
        print("❌ kb_id not provided in room metadata.")
        return

    persona = metadata.get("AGENT_PERSONA")

    userdata = UserData(kb_id=kb_id, persona=persona, ctx=ctx)

    session = AgentSession(
        userdata=userdata,
        llm=openai.LLM(model=metadata.get("model_llm")),
        tts=openai.TTS(
            model=metadata.get("model_tts"),
            voice=metadata.get("voice_tts")
        ),
        stt=deepgram.STT(
            model=metadata.get("model_stt"),
            # language=metadata.get("lang_stt", "multi"),
            language="en",
            punctuate=True,  # auto add , . 
            keyterms=["Retell", "Walmart", "PBX", "Angel"],  # more focused on these words and auto correct
        ),
        vad=silero.VAD.load(),
       
    )   

    agent = KBAgent()

    @session.on("user_input_transcribed")
    def on_transcribed(event: UserInputTranscribedEvent):
        if event.is_final:
            agent.transcript_log.append(f"User: {event.transcript}")
            print(f"[📝 Transcript] {event.transcript}")

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

   
    # ✅ Save transcript to file at shutdown
    async def write_transcript():
        try:
            session_dict = session.history.to_dict()
            chat_analysis = await analyze_chat(session_dict)
            payload = {
            "agent_id": "57e1ba83f1924ab6",
            "agent_version": 0,
            "chat_status": "ended",
            "retell_llm_dynamic_variables": {
                "customer_name": "John Doe"
            },
            "metadata": {},
            "start_timestamp": session.userdata.start_timestamp,
            "end_timestamp": int(time.time() * 1000),
            "transcript": "\n".join(
        f"{item['role'].capitalize()}: {' '.join(item['content'])}"
        for item in session.history.to_dict().get("items", [])
    ),
            "chat_cost": {
                "product_costs": [
                    {
                        "product": metadata.get("model_llm", "gpt-4o-mini"),
                        "unitPrice": 1,
                        "cost": 50
                    }
                ],
                "combined_cost": 50
            },
            "chat_analysis": chat_analysis
        }

       
            async with httpx.AsyncClient(timeout=5.0) as client:
               
                AUTH_TOKEN=os.getenv('AUTH_TOKEN')
                response = await client.post(
                    url=os.getenv("CHAT_API_URL", "http://localhost:8000/create-chat"),
                    json=payload,
                    headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
                )
                response.raise_for_status()
         
        except Exception as e:
            print(f"❌ Failed to write transcript: {e}")

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