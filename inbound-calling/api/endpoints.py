from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from schemas import schemas
from models import models
from core.database import SessionLocal
from crud import crud

router = APIRouter()

# Dependency to get DB session
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()


# ==================== SIP TRUNK ====================

@router.post("/trunks/", response_model=schemas.SIPTrunkOut)
async def create_trunk(trunk: schemas.SIPTrunkCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_trunk(db, trunk.dict())


@router.get("/trunks/", response_model=List[schemas.SIPTrunkOut])
async def list_trunks(db: AsyncSession = Depends(get_db)):
    return await crud.get_trunks(db)


# ==================== DISPATCH RULE ====================

@router.post("/dispatch_rules/", response_model=schemas.SIPDispatchRuleOut)
async def create_dispatch_rule(rule: schemas.SIPDispatchRuleCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_dispatch_rule(db, rule.dict())


@router.get("/dispatch_rules/", response_model=List[schemas.SIPDispatchRuleOut])
async def list_dispatch_rules(db: AsyncSession = Depends(get_db)):
    return await crud.get_dispatch_rules(db)


# ==================== CALL SESSION ====================

@router.get("/call_sessions/", response_model=List[schemas.CallSessionOut])
async def list_call_sessions(db: AsyncSession = Depends(get_db)):
    return await crud.get_call_sessions(db)
