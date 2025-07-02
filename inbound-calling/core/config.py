from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_SIP_URI: str
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
