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
<<<<<<< HEAD
    user = db.query(User).filter(User.email == payload.get("email")).first()
=======
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload: email not found")
    user = db.query(User).filter(User.email == email).first()
>>>>>>> 32bcc129c02ddffa8479432c89ec51cbc144831f
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user