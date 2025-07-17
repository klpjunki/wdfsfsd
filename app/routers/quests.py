from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models import Quest, User
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter(prefix="/quests", tags=["quests"])

class QuestOut(BaseModel):
    id: int
    title: str
    quest_type: str
    reward_type: str
    reward_value: int
    active: bool
    url: str | None = None
    description: str | None = None

    class Config:
        from_attributes = True

@router.get("/", response_model=list[QuestOut])
async def get_quests(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        Quest.__table__.select().where(Quest.active == True)
    )
    quests = result.fetchall()
    return [QuestOut(
        id=quest.id,
        title=quest.title,
        quest_type=quest.quest_type,
        reward_type=quest.reward_type,
        reward_value=quest.reward_value,
        active=quest.active,
        url=quest.url,  
        description=quest.description  
    ) for quest in quests]

class QuestCompleteRequest(BaseModel):
    telegram_id: int
    quest_id: int

@router.post("/complete")
async def complete_quest(data: QuestCompleteRequest, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    quest = await db.get(Quest, data.quest_id)
    if not quest or not quest.active:
        raise HTTPException(404, "Quest not found or inactive")

    # Пример: буст на 24 часа
    if quest.reward_type == "boost":
        user.boost_expiry = datetime.utcnow() + timedelta(hours=24)
    elif quest.reward_type == "energy":
        user.energy = min(user.energy + quest.reward_value, user.max_energy)
    elif quest.reward_type == "coins":
        user.coins += quest.reward_value

    await db.commit()
    return {"status": "ok", "reward_type": quest.reward_type, "reward_value": quest.reward_value}
