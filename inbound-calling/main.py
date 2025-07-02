from fastapi import FastAPI
from core.database import engine, Base

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        print("Database connected successfully on startup.")
    except Exception as e:
        print(f"Database connection failed: {e}")

@app.get("/")
async def read_root():
    return {"msg": "Async Working"}
