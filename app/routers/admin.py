from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from pydantic import BaseModel
from app.models import User
from app.routers.users import get_max_energy
from app.models import ExchangeRate
from app.schemas import  SetExchangeRateRequest
from sqlalchemy import delete
import os  
from sqlalchemy.future import select
from datetime import datetime
from sqlalchemy.orm import joinedload
from app.models import ExchangeRequest
from app.models import PromoCode
from app.schemas import PromoCodeCreate
from app.models import Quest, UserQuestStatus
from datetime import datetime, timedelta, timezone


import random
import string

router = APIRouter(prefix="", tags=["admin"]) 

def get_admin_secret() -> str:
    return os.getenv("ADMIN_SECRET", "default_secret")

def check_admin(secret: str):
    return secret == os.getenv("ADMIN_SECRET")  
class AdminCheckRequest(BaseModel):
    admin_secret: str

class QuestCreate(BaseModel):
    title: str
    quest_type: str
    reward_type: str
    reward_value: int
    url: str | None = None
    description: str | None = None

class DailyBonusConfig(BaseModel):
    day_1: int = 1000
    day_2: int = 3500
    day_3: int = 5000
    day_4: int = 7500
    day_5: int = 9000
    day_6: int = 11500
    day_7: int = 15000

bonus_config = DailyBonusConfig()  

@router.post("/daily-bonus-config")
async def update_bonus_config(
    new_config: DailyBonusConfig,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    global bonus_config
    bonus_config = new_config
    return {"status": "config_updated"}

@router.post("/check")
async def check_admin_secret(
    data: AdminCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    if data.admin_secret == os.getenv("ADMIN_SECRET"):
        return {"status": "valid"}
    raise HTTPException(403, "Forbidden")

class UpdateCoinsRequest(BaseModel):
    telegram_id: int
    amount: int

class UpdateLevelRequest(BaseModel):
    telegram_id: int
    new_level: int

class UpdateClicksRequest(BaseModel):
    telegram_id: int
    clicks: int

class UpdateEnergyRequest(BaseModel):
    telegram_id: int
    energy: int

@router.post("/add-coins", summary="Начислить монеты пользователю")
async def add_coins(
    data: UpdateCoinsRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    user.coins += data.amount
    await db.commit()
    return {"status": "success", "new_balance": user.coins}

@router.post("/remove-coins", summary="Списать монеты у пользователя")
async def remove_coins(
    data: UpdateCoinsRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    user.coins = max(0, user.coins - data.amount)
    await db.commit()
    return {"status": "success", "new_balance": user.coins}

@router.post("/set-level", summary="Установить уровень пользователя")
async def set_level(
    data: UpdateLevelRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    if data.new_level < 1:
        raise HTTPException(400, "Level cannot be less than 1")
    
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    user.level = data.new_level
    user.max_energy = get_max_energy(data.new_level)
    user.energy = min(user.energy, user.max_energy)
    
    await db.commit()
    await db.refresh(user)
    
    return {
        "status": "success",
        "new_level": user.level,
        "new_max_energy": user.max_energy,
        "current_energy": user.energy
    }

@router.post("/set-clicks", summary="Установить количество кликов")
async def set_clicks(
    data: UpdateClicksRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    user.total_clicks = max(0, data.clicks)
    await db.commit()
    return {"status": "success", "new_clicks": user.total_clicks}

@router.post("/set-energy", summary="Установить значение энергии")
async def set_energy(
    data: UpdateEnergyRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    user.energy = max(0, min(data.energy, user.max_energy))
    await db.commit()
    return {"status": "success", "new_energy": user.energy}

class UpdateRoleRequest(BaseModel):
    telegram_id: int
    new_role: str  

@router.post("/set-role", summary="Изменить роль пользователя")
async def set_role(
    data: UpdateRoleRequest,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    if data.new_role not in ["player", "youtuber"]:
        raise HTTPException(400, "Недопустимая роль")
    user = await db.get(User, data.telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    user.role = data.new_role
    await db.commit()
    await db.refresh(user)
    return {"status": "success", "telegram_id": user.telegram_id, "new_role": user.role}

@router.post("/set-exchange-rate")
async def set_exchange_rate(
    rate_data: SetExchangeRateRequest,
    db: AsyncSession = Depends(get_db),
    admin_secret: str = Depends(get_admin_secret)
):
    await db.execute(
        delete(ExchangeRate)
        .where(ExchangeRate.to_currency == rate_data.currency.value)
    )
    
    new_rate = ExchangeRate(
        from_currency="COIN",
        to_currency=rate_data.currency.value,
        rate=rate_data.rate
    )
    db.add(new_rate)
    await db.commit()
    
    return {"status": "success", "new_rate": rate_data.rate}

@router.get("/exchange-requests")
async def get_exchange_requests(
    db: AsyncSession = Depends(get_db),
    admin_secret: str = Depends(get_admin_secret)
):
    result = await db.execute(
        select(ExchangeRequest)
        .options(joinedload(ExchangeRequest.user))
        .order_by(ExchangeRequest.created_at.desc())
    )
    return result.scalars().unique().all()

@router.post("/exchange-requests/{request_id}/approve")
async def approve_exchange(
    request_id: int,
    db: AsyncSession = Depends(get_db),
    admin_secret: str = Depends(get_admin_secret)
):
    request = await db.get(ExchangeRequest, request_id)
    if not request:
        raise HTTPException(404, "Заявка не найдена")
    
    if request.status != "pending":
        raise HTTPException(400, "Заявка уже обработана")
    
    user = await db.get(User, request.user_id)
    if not user:
        raise HTTPException(404, "Пользователь не найден")
    
    user.locked_coins -= request.amount

    if request.to_currency == "Танки Блитц":
        user.tanki_blitz_balance += request.received_amount
    elif request.to_currency == "Мир танков":
        user.mir_tankov_balance += request.received_amount
    elif request.to_currency == "Wot Blitz":
        user.wot_blitz_balance += request.received_amount
    
    request.status = "approved"
    request.processed_at = datetime.now(timezone.utc)
    
    await db.commit()
    return {"status": "success"}

@router.post("/exchange-requests/{request_id}/reject")
async def reject_exchange(
    request_id: int,
    db: AsyncSession = Depends(get_db),
    admin_secret: str = Depends(get_admin_secret)
):
    request = await db.get(ExchangeRequest, request_id)
    if not request:
        raise HTTPException(404, "Заявка не найдена")
    
    if request.status != "pending":
        raise HTTPException(400, "Заявка уже обработана")
    
    user = await db.get(User, request.user_id)
    if not user:
        raise HTTPException(404, "Пользователь не найден")
    
    user.coins += request.amount
    user.locked_coins -= request.amount
    
    request.status = "rejected"
    request.processed_at = datetime.now(timezone.utc)

    
    await db.commit()
    return {"status": "success"}

def generate_promocode(length: int = 10) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

@router.post("/create-promocode", summary="Создать промокод")
async def create_promocode(
    data: PromoCodeCreate,
    admin_secret: str,
    db: AsyncSession = Depends(get_db)
):
    if not check_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    code = data.code or generate_promocode()
    
    result = await db.execute(select(PromoCode).where(PromoCode.code == code))
    if result.scalar_one_or_none():
        raise HTTPException(400, "Промокод уже существует")
    
    promocode = PromoCode(
        code=code,
        reward_type=data.reward_type,
        value=data.value,
        uses_left=data.uses_left,
    )
    db.add(promocode)
    await db.commit()
    await db.refresh(promocode)
    return {
        "status": "success",
        "code": promocode.code,
        "reward_type": promocode.reward_type,
        "value": promocode.value,
        "uses_left": promocode.uses_left,
        "expiry": promocode.expiry
    }

@router.get("/exchange-rates")
async def get_exchange_rates(
    db: AsyncSession = Depends(get_db),
    admin_secret: str = Depends(get_admin_secret)
):
    result = await db.execute(
        select(ExchangeRate)
        .order_by(ExchangeRate.to_currency)
    )
    rates = result.scalars().all()
    return [
        {
            "to_currency": rate.to_currency,
            "rate": rate.rate,
            "last_updated": rate.last_updated
        }
        for rate in rates
    ]
# === НОВОЕ: CRUD для динамических квестов ===
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.database import get_db
from app.models import Quest
from app.schemas import QuestUpsert, QuestOut

# если у тебя уже есть router = APIRouter(...), используй его
# иначе раскомментируй строку ниже:
# router = APIRouter()

def _is_admin(secret: str) -> bool:
    return secret == os.getenv("ADMIN_SECRET")

@router.post("/quests", response_model=QuestOut)
async def create_quest(payload: QuestUpsert, admin_secret: str, db: AsyncSession = Depends(get_db)):
    if not _is_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    quest = Quest(
        title=payload.title,
        quest_type=payload.quest_type,
        url=payload.url,
        reward_type=payload.reward_type,
        reward_value=payload.reward_value,
        description=payload.description,
        active=payload.active,
    )
    db.add(quest)
    await db.commit()
    await db.refresh(quest)
    return quest

@router.get("/quests", response_model=list[QuestOut])
async def list_quests(admin_secret: str, db: AsyncSession = Depends(get_db)):
    if not _is_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    res = await db.execute(select(Quest).order_by(Quest.id.desc()))
    return list(res.scalars())

@router.patch("/quests/{quest_id}", response_model=QuestOut)
async def update_quest(quest_id: int, payload: QuestUpsert, admin_secret: str, db: AsyncSession = Depends(get_db)):
    if not _is_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    res = await db.execute(select(Quest).where(Quest.id == quest_id))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found")

    quest.title = payload.title
    quest.quest_type = payload.quest_type
    quest.url = payload.url
    quest.reward_type = payload.reward_type
    quest.reward_value = payload.reward_value
    quest.description = payload.description
    quest.active = payload.active

    await db.commit()
    await db.refresh(quest)
    return quest

@router.delete("/quests/{quest_id}")
async def delete_quest(quest_id: int, admin_secret: str, db: AsyncSession = Depends(get_db)):
    if not _is_admin(admin_secret):
        raise HTTPException(403, "Forbidden")
    
    try:
        # СНАЧАЛА удаляем все связанные записи в user_quest_status
        print(f"🗑️ Удаляем связанные записи для квеста {quest_id}")
        result = await db.execute(
            delete(UserQuestStatus).where(UserQuestStatus.quest_id == quest_id)
        )
        print(f"✅ Удалено {result.rowcount} записей из user_quest_status")
        
        # ПОТОМ удаляем сам квест
        print(f"🗑️ Удаляем квест {quest_id}")
        quest_result = await db.execute(delete(Quest).where(Quest.id == quest_id))
        
        if quest_result.rowcount == 0:
            raise HTTPException(404, "Quest not found")
            
        print(f"✅ Квест {quest_id} успешно удален")
        await db.commit()
        
        return {"status": "ok", "message": f"Quest {quest_id} deleted successfully"}
        
    except Exception as e:
        await db.rollback()
        print(f"❌ Ошибка при удалении квеста {quest_id}: {str(e)}")
        raise HTTPException(500, f"Failed to delete quest: {str(e)}")