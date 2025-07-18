from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config.settings import DATABASE_URL
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

# Base configuration
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from models.models import User
    from utils.security import decode_token

    payload = decode_token(token)
    user = db.query(User).filter(User.email == payload.get("email")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user 