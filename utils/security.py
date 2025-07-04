from fastapi import HTTPException
from jose import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from config.settings import SECRET_KEY, ALGORITHM
from datetime import datetime, timedelta
import os
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

# def create_token(data: dict):
#     return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def create_token(data: dict):
    expire = int(os.getenv("TOKEN_EXPIRE_MINUTES", 60))
    payload = {
        "username": data["username"],
        "exp": expire
    }
    token = jwt.encode(payload, os.getenv("SECRET_KEY"), algorithm="HS256")
    return token, expire 



def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
