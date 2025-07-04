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

    print(f"Token: {token}")  # Debug
    payload = decode_token(token)
    print(f"Payload: {payload}")  # Debug
    username = payload.get("username")
    print(f"Username: {username}")  # Debug
    user = db.query(User).filter(User.username == username).first()
    if not user:
        print("User not found in database")  # Debug
        raise HTTPException(status_code=401, detail="User not found")
    print(f"User found: {user.username}")  # Debug
    return user