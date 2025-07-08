from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any


class UserSignup(BaseModel):
    username: str = Field(..., min_length=5)
    email: EmailStr
    full_name: str
    password: str
    retall_api_key: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UpdateUser(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class WorkspaceOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: str
    owner_username: str

    class Config:
        from_attributes = True

class InviteMember(BaseModel):
    username: str
    role: Optional[str] = "member"

class DispatchRequest(BaseModel):
    room_name: str
    agent_name: str
    metadata: dict
    
class CreateRoomRequestSchema(BaseModel):
    agent_name: str
    room_name: str
    participant_identity: str
    participant_name: str
    metadata: Optional[Dict[str, Any]] = None
    
class KnowledgeBaseCreate(BaseModel):
    name: str
    workspace_id: int

class TokenRequest(BaseModel):
    room_name: str
    user_id: str
    agent_name: str

class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    file_path: str
    created_at: datetime
    workspace_id: int

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class KnowledgeFileOut(BaseModel):
    id: int
    filename: str
    file_path: str
    kb_id: str
    uploaded_at: datetime
    extract_data: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class APIKeyCreate(BaseModel):
    name: str

class WorkspaceSettingsUpdate(BaseModel):
    default_voice: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None

# Agent schemas----------------------------------------------

class ResponseEngine(BaseModel):
    type: Literal["pbx-llm", "custom-llm", "conversation-flow"]
    llm_id: Optional[str] = None
    version: Optional[int] = None
    llm_websocket_url: Optional[str] = None

    @field_validator("llm_id")
    @classmethod
    def validate_llm_id(cls, v, info):
        if info.data.get("type") == "pbx-llm" and not v:
            raise ValueError("llm_id is required for pbx-llm type")
        return v

    @field_validator("llm_websocket_url")
    @classmethod
    def validate_ws_url(cls, v, info):
        if info.data.get("type") == "custom-llm" and not v:
            raise ValueError("llm_websocket_url is required for custom-llm type")
        return v

class VoicemailAction(BaseModel):
    type: str
    text: str

class VoicemailOption(BaseModel):
    action: VoicemailAction

class Pronunciation(BaseModel):
    word: str
    alphabet: str
    phoneme: str
    
class PhoneNumberOut(BaseModel):
    id: str
    phone_number: str
    phone_number_type: str
    phone_number_pretty: str
    inbound_agent_id: str
    outbound_agent_id: str
    inbound_agent_version: str
    outbound_agent_version: str
    area_code: str
    nickname: str
    inbound_webhook_url: str
    last_modification_timestamp: str
class PostCallAnalysis(BaseModel):
    type: str
    name: str
    description: str
    examples: List[str]

class UserDTMFOptions(BaseModel):
    digit_limit: int
    termination_key: str
    timeout_ms: int

class AgentCreate(BaseModel):
    workspace_id: int
    name: Optional[str] = None
    version: Optional[int] = 0
    voice_id: str
    voice_model: Optional[str] = None
    fallback_voice_ids: Optional[List[str]] = None
    voice_temperature: Optional[float] = 1.0
    voice_speed: Optional[float] = 1.0
    volume: Optional[float] = 1.0
    responsiveness: Optional[float] = 1.0
    interruption_sensitivity: Optional[float] = 1.0
    enable_backchannel: Optional[bool] = False
    backchannel_frequency: Optional[float] = 0.8
    backchannel_words: Optional[List[str]] = None
    reminder_trigger_ms: Optional[int] = 10000
    reminder_max_count: Optional[int] = 1
    ambient_sound: Optional[str] = None
    ambient_sound_volume: Optional[float] = 1.0
    language: Optional[str] = "en-US"
    webhook_url: Optional[str] = None
    boosted_keywords: Optional[List[str]] = None
    opt_out_sensitive_data_storage: Optional[bool] = False
    opt_in_signed_url: Optional[bool] = True
    pronunciation_dictionary: Optional[List[Pronunciation]] = None
    normalize_for_speech: Optional[bool] = True
    end_call_after_silence_ms: Optional[int] = 600000
    max_call_duration_ms: Optional[int] = 3600000
    voicemail_option: Optional[VoicemailOption] = None
    post_call_analysis_data: Optional[List[PostCallAnalysis]] = None
    post_call_analysis_model: Optional[str] = "gpt-4o-mini"
    begin_message_delay_ms: Optional[int] = 1000
    ring_duration_ms: Optional[int] = 30000
    stt_mode: Optional[str] = "fast"
    vocab_specialization: Optional[str] = "general"
    allow_user_dtmf: Optional[bool] = True
    user_dtmf_options: Optional[UserDTMFOptions] = None
    denoising_mode: Optional[str] = "noise-cancellation"
    response_engine: ResponseEngine

class AgentOut(AgentCreate):
    agent_id: str = Field(..., alias="id")  # ← maps DB's `id` to `agent_id`
    version: Optional[int] = 0
    is_published: Optional[bool]
    last_modification_timestamp: Optional[int]

    class Config:
        from_attributes = True  # for Pydantic v2
        validate_by_name = True


# PBX LLM SCHEMAS----------------------------------------------

class Tool(BaseModel):
    type: str
    name: str
    description: str

class StateEdge(BaseModel):
    destination_state_name: str
    description: str

class LLMState(BaseModel):
    name: str
    state_prompt: str
    edges: Optional[List[StateEdge]] = []
    tools: Optional[List[Tool]] = []

class KnowledgeFileOut(BaseModel):
    id: int
    # kb_id: str
    # filename: str
    # file_path: Optional[str]
    # extract_data: Optional[str]
    status: str
    source_type: str
    embedding: Optional[List[float]]
    uploaded_at: datetime

    class Config:
        from_attributes = True

class StartAgentRequest(BaseModel):
    room: str
    kb_id: str
    persona: str = "You are a helpful voice assistant."
    model_llm: str = "gpt-4o"
    model_tts: str = "tts-1"
    voice_tts: str = "nova"
    model_stt: str = "general"

class PBXLLMCreate(BaseModel):
    workspace_id: int
    version: Optional[int] = 0
    model: Optional[str] = None
    s2s_model: Optional[str] = None
    model_temperature: Optional[float] = 0.0
    model_high_priority: Optional[bool] = False
    tool_call_strict_mode: Optional[bool] = False
    general_prompt: Optional[str] = None
    general_tools: Optional[List[Dict[str, Any]]] = None
    states: Optional[List[Dict[str, Any]]] = None
    starting_state: Optional[str] = None
    begin_message: Optional[str] = None
    default_dynamic_variables: Optional[Dict[str, str]] = None
    knowledge_base_ids: Optional[List[str]] = None

class PBXLLMOut(BaseModel):
    workspace_id: int
    version: int
    model: str
    s2s_model: str
    model_temperature: float
    model_high_priority: bool
    tool_call_strict_mode: bool
    general_prompt: str
    general_tools: List[Tool]
    begin_message: str
    default_dynamic_variables: Dict[str, Any]
    knowledge_base_ids: List[int]

    class Config:
        from_attributes = True

#  Chat room -----------------------------------------------

class MessageWithToolCall(BaseModel):
    message_id: str
    role: str
    content: str
    created_timestamp: int


class ProductCost(BaseModel):
    product: str
    unitPrice: float
    cost: float


class ChatCost(BaseModel):
    product_costs: List[ProductCost]
    combined_cost: float


class ChatAnalysis(BaseModel):
    chat_summary: str
    user_sentiment: str
    chat_successful: bool
    custom_analysis_data: Optional[Dict[str, Any]] = None


class CreateChatRequest(BaseModel):
    agent_id: str
    chat_status: str
    agent_version: Optional[int] = 0
    metadata: Optional[Dict[str, Any]] = None
    retell_llm_dynamic_variables: Optional[Dict[str, str]] = None
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    transcript: Optional[str] = None
    chat_cost: Optional[ChatCost] = None
    chat_analysis: Optional[ChatAnalysis] = None


class CreateChatResponse(BaseModel):
    chat_id: str
    agent_id: str
    chat_status: str
    retell_llm_dynamic_variables: Optional[Dict[str, str]] = None
    collected_dynamic_variables: Optional[Dict[str, str]] = None
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    transcript: Optional[str] = None
    message_with_tool_calls: Optional[List[MessageWithToolCall]] = None
    metadata: Optional[Dict[str, Any]] = None
    chat_cost: Optional[ChatCost] = None
    chat_analysis: Optional[ChatAnalysis] = None

# voice

class VoiceBase(BaseModel):
    voice_id: str
    voice_name: str
    provider: Literal["elevenlabs", "openai", "deepgram"]
    gender: Literal["male", "female"]
    accent: Optional[str] = None
    age: Optional[str] = None
    preview_audio_url: Optional[str] = None

class VoiceCreate(VoiceBase):
    pass

class VoiceOut(BaseModel):
    voice_id: str
    voice_name: str
    provider: str
    gender: str
    accent: str
    age: str
    preview_audio_url: str

    class Config:
        from_attributes = True
    

class VoiceListResponse(BaseModel):
    status: bool
    data: List[VoiceOut]

#import phone
class PhoneNumberCreate(BaseModel):
    phone_number: str
    phone_number_type: str
    sip_trunk_auth_username: Optional[str] = None
    sip_trunk_auth_password: Optional[str] = None
    inbound_agent_id: Optional[str] = None
    outbound_agent_id: Optional[str] = None
    inbound_agent_version: Optional[int] = None
    outbound_agent_version: Optional[int] = None
    nickname: Optional[str] = None
    inbound_webhook_url: Optional[str] = None