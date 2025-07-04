from sqlalchemy.orm import Session
from models import LLMVoice
from ..database import SessionLocal

def populate_db():
    # Sample data
    voice_data = {
        "voice_id": "openm325mmn",
        "voice_name": "James2",
        "provider": "openai",
        "gender": "male",
        "accent": "american",
        "age": "32",
        "preview_audio_url": "https://example.com/audio/james-preview.mp3"
    }

    # Create a new database session
    db = SessionLocal()
    try:
        # Check if the voice_id already exists to avoid duplicates
        existing_voice = db.query(LLMVoice).filter(LLMVoice.voice_id == voice_data["voice_id"]).first()
        if not existing_voice:
            # Create a new voice entry
            new_voice = LLMVoice(**voice_data)
            db.add(new_voice)
            db.commit()
            print("Voice added successfully!")
        else:
            print("Voice with this voice_id already exists!")
    finally:
        db.close()

if __name__ == "__main__":
    populate_db()