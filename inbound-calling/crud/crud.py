from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import models


# ============== SIP Trunk ==============

async def create_trunk(db: AsyncSession, trunk_data: dict):
    db_trunk = models.SIPTrunk(**trunk_data)
    db.add(db_trunk)
    await db.commit()
    await db.refresh(db_trunk)
    return db_trunk


async def get_trunks(db: AsyncSession):
    result = await db.execute(select(models.SIPTrunk))
    return result.scalars().all()


# ============== Dispatch Rule ==============

async def create_dispatch_rule(db: AsyncSession, rule_data: dict):
    db_rule = models.SIPDispatchRule(**rule_data)
    db.add(db_rule)
    await db.commit()
    await db.refresh(db_rule)
    return db_rule


async def get_dispatch_rules(db: AsyncSession):
    result = await db.execute(select(models.SIPDispatchRule))
    return result.scalars().all()


# ============== Call Sessions ==============

async def get_call_sessions(db: AsyncSession):
    result = await db.execute(select(models.CallSession))
    return result.scalars().all()


async def create_call_session(db: AsyncSession, session_data: dict):
    db_session = models.CallSession(**session_data)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session
