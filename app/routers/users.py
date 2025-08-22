from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import User, PromoCode
from pydantic import BaseModel,  field_validator
import random
from typing import Optional
import logging
from datetime import datetime, timedelta, timezone
import string  
from app.schemas import UserCreate
from app.schemas import UserCreate
from app.models import ExchangeRate
from app.schemas import ExchangeRequestCreate
from sqlalchemy import delete
import os  
from sqlalchemy.orm import Session
from app.models import UserQuestStatus, Quest, User
from app.models import User, PromoCode, UserPromoCode



from app.schemas import ExchangeRequestCreate  
from app.models import ExchangeRequest  

router = APIRouter(tags=["users"])


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

class UserOut(BaseModel):
    telegram_id: int
    username: Optional[str] = None
    coins: int = 0
    energy: int = 6500
    max_energy: int = 6500
    level: int = 1
    total_clicks: int = 0
    referral_code: Optional[str] = None
    referred_by: Optional[str] = None
    milestone_5_friends_claimed: bool = False
    reward_5_friends_claimed: bool = False
    reward_10_friends_claimed: bool = False
    daily_streak: int = 0
    last_daily_login: Optional[datetime] = None
    role: str = "player"
    boost_expiry: Optional[datetime] = None
    boost_multiplier: int = 1
    seconds_left: int = 0

    @field_validator('coins', 'energy', 'max_energy', 'level', 'total_clicks', mode='before')
    @classmethod
    def handle_numeric_nulls(cls, value):
        return value if value is not None else 0

    @field_validator(
    'milestone_5_friends_claimed', 'reward_5_friends_claimed',
    'reward_10_friends_claimed', mode='before'
    )
    @classmethod
    def handle_bool_nulls(cls, value):
        return value if value is not None else False

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}
rs = {datetime: lambda dt: dt.isoformat()}

class ClickResponse(BaseModel):
    coins: int
    energy: int
    total_clicks: int

class ClickRequest(BaseModel):
    extra_data: dict = None


class TransferByUsernameRequest(BaseModel):
    sender_id: int
    receiver_username: str
    amount: int

    @field_validator('receiver_username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Имя пользователя не может быть пустым")
        return v.strip()

GLOBAL_MARKET_TIMER_START = datetime.now(timezone.utc)
GLOBAL_MARKET_TIMER_END = datetime(2025, 10, 17, 16, 0, 0, tzinfo=timezone.utc)


def generate_referral_code(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

async def generate_unique_referral_code(db: AsyncSession) -> str:
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        result = await db.execute(select(User).where(User.referral_code == code))
        if not result.scalar_one_or_none():
            return code

def get_max_energy(level: int) -> int:
    return 6500 + 500 * (min(level, 10) - 1)

def calculate_energy_regeneration(user):
    try:
        if not hasattr(user, 'last_energy_update') or not user.last_energy_update:
            user.last_energy_update = datetime.now()
            return user.energy

        now = datetime.now()
        time_diff = (now - user.last_energy_update).total_seconds()
        
        if time_diff < 0:
            user.last_energy_update = now
            return user.energy

        energy_restored = int(time_diff / 6)
        base_max_energy = get_max_energy(user.level)
        milestone_claimed = bool(getattr(user, 'milestone_5_friends_claimed', False))
        
        if energy_restored > 0 and user.energy < base_max_energy:
            new_energy = min(
                user.energy + energy_restored, 
                base_max_energy + 500 if milestone_claimed else base_max_energy
            )
            user.energy = new_energy
            user.last_energy_update = now

        return user.energy
        
    except Exception as e:
        logging.error(f"Error in energy regen: {str(e)}")
        return user.energy

async def check_and_award_milestone_bonus(db: AsyncSession, user: User):
    milestone_claimed = bool(getattr(user, 'milestone_5_friends_claimed', 0))
    
    if milestone_claimed:
        return False
    
    result = await db.execute(
        select(User).where(User.referred_by == user.referral_code)
    )
    referrals = result.scalars().all()
    
    if len(referrals) >= 5:
        user.energy = int(user.energy) + 500
        user.milestone_5_friends_claimed = True
        
        await db.commit()
        await db.refresh(user)
        return True
    
    return False
@router.get("/check_referral/{referral_code}")
async def check_referral_code(referral_code: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.referral_code == referral_code.upper())
    )
    referrer = result.scalar_one_or_none()
    
    if not referrer:
        raise HTTPException(status_code=404, detail="Referral code not found")
    
    return {
        "valid": True,
        "referrer_username": referrer.username,
        "referrer_level": referrer.level
    }
def is_boost_active(user):
    if not user.boost_expiry:
        return False
    now = datetime.now(timezone.utc)
    boost_expiry = user.boost_expiry.replace(tzinfo=timezone.utc) if user.boost_expiry.tzinfo is None else user.boost_expiry.astimezone(timezone.utc)
    return boost_expiry > now
@router.post("/{telegram_id}/claim_referral_reward")
async def claim_referral_reward(telegram_id: int, quest_type: str, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    result = await db.execute(
        select(User).where(User.referred_by == user.referral_code)
    )
    referrals = result.scalars().all()
    referral_count = len(referrals)
    if quest_type == "5_friends":
        if referral_count < 5:
            raise HTTPException(status_code=400, detail="Недостаточно приглашенных друзей")
        
        if getattr(user, 'reward_5_friends_claimed', False):
            raise HTTPException(status_code=400, detail="Награда уже получена")
        user.energy += 6500
        user.reward_5_friends_claimed = True
        
        await db.commit()
        
        return {
            "status": "success",
            "message": "Получена награда за приглашение 5 друзей!",
            "reward": {
                "type": "energy",
                "amount": 6500
            },
            "current_energy": user.energy
        }
    elif quest_type == "10_friends":
        if referral_count < 10:
            raise HTTPException(status_code=400, detail="Недостаточно приглашенных друзей")
        
        if getattr(user, 'reward_10_friends_claimed', False):
            raise HTTPException(status_code=400, detail="Награда уже получена")
        user.energy += 12000
        user.reward_10_friends_claimed = True
        
        await db.commit()
        
        return {
            "status": "success",
            ""
            "": "",
            "reward": {
                "type": "energy",
                "amount": 12000
            },
            "current_energy": user.energy
        }
    
    else:
        raise HTTPException(status_code=400, detail="Неизвестный тип задания")
    
@router.get("/public-exchange-rates")
async def public_exchange_rates(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ExchangeRate).order_by(ExchangeRate.to_currency))
    rates = result.scalars().all()
    return {
        "rates": [
            {"to_currency": rate.to_currency, "rate": rate.rate}
            for rate in rates
        ]
    }


@router.get("/{telegram_id}", response_model=UserOut)
async def get_user(
    telegram_id: int,
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    now = datetime.now(timezone.utc)
    seconds_left = max(0, int((GLOBAL_MARKET_TIMER_END - now).total_seconds())) if user.role == "player" else 0

    return {
    "telegram_id": user.telegram_id,
    "username": user.username,
    "coins": user.coins,
    "level": user.level,
    "energy": user.energy,
    "max_energy": user.max_energy,
    "total_clicks": user.total_clicks,
    "referral_code": user.referral_code,
    "referred_by": user.referred_by,
    "milestone_5_friends_claimed": user.milestone_5_friends_claimed,
    "reward_5_friends_claimed": user.reward_5_friends_claimed,
    "reward_10_friends_claimed": user.reward_10_friends_claimed,
    "daily_streak": user.daily_streak,
    "last_daily_login": user.last_daily_login,
    "role": user.role,
    "boost_expiry": user.boost_expiry,
    "boost_multiplier": user.boost_multiplier,
    "seconds_left": seconds_left
}


@router.post("/{telegram_id}/click", response_model=ClickResponse)
async def click_coin(
    telegram_id: int, 
    click_data: ClickRequest = ClickRequest(),
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        calculate_energy_regeneration(user)
        
        if user.energy < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough energy. Current: {user.energy}, Required: 10"
            )
        
        now = datetime.now(timezone.utc)
        boost_active = False
        if user.boost_expiry:
            boost_expiry_utc = user.boost_expiry.astimezone(timezone.utc) if user.boost_expiry.tzinfo else user.boost_expiry.replace(tzinfo=timezone.utc)
            boost_active = boost_expiry_utc > now
        coins_per_click = user.boost_multiplier if (boost_active and user.boost_multiplier) else 1
        user.energy = max(user.energy - 10, 0)
        user.coins += coins_per_click
        user.total_clicks += 1
        old_level = user.level
        new_level = min(10, 1 + (user.total_clicks // 10000))  

        if new_level != old_level:
            user.level = new_level
            user.max_energy = get_max_energy(new_level)
            user.energy = min(user.energy, user.max_energy)

        await db.commit()
        await db.refresh(user)

        return ClickResponse(
            coins=user.coins,
            energy=user.energy,
            total_clicks=user.total_clicks
        )

    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal error: {str(e)}"
        )

@router.get("/{telegram_id}/referrals")
async def get_referrals(telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        result = await db.execute(
            select(User).where(User.referred_by == user.referral_code)
        )
        referrals = result.scalars().all()
        referral_count = len(referrals)
        milestone_claimed = bool(getattr(user, 'milestone_5_friends_claimed', 0))
        
        milestone_status = {
            "eligible": referral_count >= 5,
            "claimed": milestone_claimed,
            "progress": f"{referral_count}/5",
            "remaining": max(0, 5 - referral_count)
        }
        
        return {
            "referral_code": user.referral_code,
            "referral_count": referral_count,
            "milestone_5_friends": milestone_status,
            "referrals": [
                {
                    "telegram_id": ref.telegram_id,
                    "username": ref.username,
                    "level": ref.level,
                    "total_clicks": ref.total_clicks,
                    "coins": ref.coins
                } for ref in referrals
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error")
@router.post("/", response_model=UserOut)
async def create_user(
    user_data: UserCreate, 
    db: AsyncSession = Depends(get_db)
):
    existing_user = await db.get(User, user_data.telegram_id)
    if existing_user:
        return existing_user
    referral_code = await generate_unique_referral_code(db)
    role = user_data.role if user_data.role else "player"
    market_timer_end = (
        datetime.now(timezone.utc) + timedelta(days=90) 
        if role == "player" 
        else None
    )
    user = User(
        telegram_id=user_data.telegram_id,
        username=user_data.username,
        coins=0,
        energy=6500,
        max_energy=6500,
        level=1,
        total_clicks=0,
        referral_code=referral_code,
        referred_by=user_data.referred_by,
        boost_expiry=None,
        boost_multiplier=1,
        role=user_data.role or "player",
        last_energy_update=datetime.now(timezone.utc)
                )

    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user



@router.post("/transfer_by_username")
async def transfer_by_username(
    data: TransferByUsernameRequest,
    db: AsyncSession = Depends(get_db)
):
    sender = await db.get(User, data.sender_id)
    if not sender:
        raise HTTPException(404, "Sender not found")
    result = await db.execute(select(User).where(User.username == data.receiver_username))
    receiver = result.scalar_one_or_none()
    if not receiver:
        raise HTTPException(404, "Receiver not found")
    if sender.coins < data.amount or data.amount <= 0:
        raise HTTPException(400, "Not enough coins or invalid amount")
    sender.coins -= data.amount
    receiver.coins += data.amount
    await db.commit()
    await db.refresh(sender)
    await db.refresh(receiver)
    return {"status": "success", "sender_coins": sender.coins, "receiver_coins": receiver.coins}



@router.post("/{telegram_id}/promocode")
async def apply_promocode(telegram_id: int, code: str, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    result = await db.execute(select(PromoCode).where(PromoCode.code == code))
    promocode = result.scalar_one_or_none()
    if not promocode or promocode.uses_left <= 0:
        raise HTTPException(400, "Invalid or expired promo code")
    
    existing = await db.execute(
        select(UserPromoCode).where(
            UserPromoCode.user_id == user.telegram_id,
            UserPromoCode.code == code
        )
    )
    used_record = existing.scalar_one_or_none()
    if used_record:
        raise HTTPException(400, "You have already used this promo code")

    if promocode.reward_type == "coins":
        user.coins += promocode.value
    elif promocode.reward_type == "energy":
        user.energy = user.energy + promocode.value
    
    promocode.uses_left -= 1

    new_use = UserPromoCode(user_id=user.telegram_id, code=code, used_at=datetime.now(timezone.utc))
    db.add(new_use)

    await db.commit()
    await db.refresh(user)
    await db.refresh(promocode)

    return {"status": "success", "reward_type": promocode.reward_type, "reward_value": promocode.value}
@router.get("/", response_model=list[UserOut])
async def get_all_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return users

@router.delete("/{telegram_id}")
async def delete_user(telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)
    await db.commit()
    return {"status": "deleted"}




class PaymentException(HTTPException):
    def __init__(self, amount: float):
        super().__init__(
            status_code=402,
            detail=f"Payment failed for amount ${amount:.2f}"
        )


@router.post("/{telegram_id}/daily-bonus")
async def claim_daily_bonus(
    telegram_id: int,
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    now = datetime.now(timezone.utc)
    last_login = user.last_daily_login

    if last_login:
        days_since_last = (now.date() - last_login.date()).days
        if days_since_last == 0:
            return {"status": "already_claimed"}
        if days_since_last > 1:
            user.level = 1
            user.daily_streak = 0
            user.total_clicks = 0
            user.energy = 6500
            user.max_energy = 6500

    user.daily_streak += 1
    bonus = user.level * 1000
    user.coins += bonus
    user.last_daily_login = now

    if user.daily_streak == 7:
        user.level = 1
        user.daily_streak = 0
        user.total_clicks = 0
        user.energy = 6500
        user.max_energy = 6500

    await db.commit()
    return {
        "streak": user.daily_streak,
        "bonus": bonus,
        "level": user.level,
        "next_day_bonus": user.level * 1000 if user.daily_streak != 0 else 1000
    }


# === НОВОЕ: выполнение динамических квестов (youtube/telegram) ===
import os
import re
import httpx
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import User, Quest, UserQuestStatus
from app.schemas import QuestOut, StartQuestRequest, ClaimQuestRequest

YOUTUBE_WAIT = timedelta(minutes=10)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def _normalize_channel_ref(raw: str) -> str:
    raw = raw.strip()
    if re.match(r"^-?\d+$", raw):
        return raw  # numeric chat id
    m = re.search(r"(?:t\.me/)?(@?[A-Za-z0-9_]+)$", raw)
    if m:
        val = m.group(1)
        if not val.startswith("@") and not val.startswith("-"):
            val = f"@{val}"
        return val
    if not raw.startswith("@") and not raw.startswith("-"):
        return f"@{raw}"
    return raw


async def _check_tg_subscription(user_telegram_id: int, channel_ref: str) -> bool:
    """
    True если пользователь состоит в канале/чате.
    БОТ ДОЛЖЕН БЫТЬ АДМИНОМ в этом канале/супергруппе.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "TELEGRAM_BOT_TOKEN not set")
    chat_id = _normalize_channel_ref(channel_ref)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChatMember"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params={"chat_id": chat_id, "user_id": user_telegram_id})
        data = r.json()
        if not data.get("ok"):
            return False
        status = data["result"]["status"]
        return status in ("member", "administrator", "creator")


@router.get("/quests/dynamic", response_model=list[QuestOut])
async def list_dynamic_quests(telegram_id: int, db: AsyncSession = Depends(get_db)):
    """
    Список активных динамических квестов (YouTube/Telegram).
    Статичные квесты по друзьям здесь НЕ затрагиваем.
    """
    res = await db.execute(select(Quest).where(Quest.active == True))
    return list(res.scalars())


@router.post("/quests/start")
async def start_quest(body: StartQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    res = await db.execute(select(Quest).where(Quest.id == body.quest_id, Quest.active == True))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found or inactive")

    # найдём/создадим статус
    res_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    us = res_us.scalar_one_or_none()
    if not us:
        us = UserQuestStatus(user_id=telegram_id, quest_id=quest.id)
        db.add(us)

    # для youtube — фиксируем старт таймера
    if quest.quest_type == "youtube":
        us.timer_started_at = datetime.now(timezone.utc)

    await db.commit()
    return {"status": "started"}


@router.post("/quests/claim")
async def claim_quest(body: ClaimQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    res = await db.execute(select(Quest).where(Quest.id == body.quest_id, Quest.active == True))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found or inactive")

    res_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    us = res_us.scalar_one_or_none()
    if not us:
        raise HTTPException(400, "Quest not started")

    if us.reward_claimed:
        return {"status": "already_claimed"}

    # проверяем условия
    if quest.quest_type == "youtube":
        if not us.timer_started_at:
            raise HTTPException(400, "Quest not started")
        if datetime.now(timezone.utc) < us.timer_started_at + YOUTUBE_WAIT:
            seconds_left = int((us.timer_started_at + YOUTUBE_WAIT - datetime.now(timezone.utc)).total_seconds())
            raise HTTPException(400, f"Wait {seconds_left} seconds")

    elif quest.quest_type == "telegram":
        ok = await _check_tg_subscription(user_telegram_id=telegram_id, channel_ref=quest.url)
        if not ok:
            raise HTTPException(400, "Подпишитесь на канал и попробуйте снова")

    # выдаём награду
    if quest.reward_type == "coins":
        user.coins += int(quest.reward_value)
    elif quest.reward_type == "energy":
        user.energy = min(user.max_energy, user.energy + int(quest.reward_value))

    us.completed = True
    us.reward_claimed = True

    await db.commit()
    return {"status": "claimed", "reward": quest.reward_type, "amount": quest.reward_value}

from sqlalchemy.future import select
from app.models import ExchangeRate

async def get_exchange_rate(db: AsyncSession, to_currency: str) -> float:
    result = await db.execute(
        select(ExchangeRate.rate)
        .where(ExchangeRate.to_currency == to_currency)
        .order_by(ExchangeRate.last_updated.desc())
        .limit(1)
    )
    rate = result.scalar_one_or_none()
    return rate if rate is not None else 1.0





@router.post("/{telegram_id}/exchange")
async def exchange_currency(
    telegram_id: int,
    exchange_data: ExchangeRequestCreate,
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.locked_coins is None:
        user.locked_coins = 0
    
    rate = await get_exchange_rate(db, exchange_data.to_currency.value)
    
    available_coins = user.coins - user.locked_coins
    if available_coins < exchange_data.amount:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough coins. Available: {available_coins}, Requested: {exchange_data.amount}"
        )
    
    converted_amount = int(exchange_data.amount * rate)
    
    user.coins -= exchange_data.amount
    user.locked_coins += exchange_data.amount
    
    exchange_request = ExchangeRequest(
        user_id=telegram_id,
        from_currency="COIN",
        to_currency=exchange_data.to_currency.value,
        amount=exchange_data.amount,
        received_amount=converted_amount,
        uid=exchange_data.uid,  
        status="pending"
    )
    
    db.add(exchange_request)
    await db.commit()
    await db.refresh(exchange_request)
    
    return {
        "status": "pending",
        "request_id": exchange_request.id,
        "message": "Заявка создана. Ожидает подтверждения администратора.",
        "frozen_coins": user.locked_coins
    }





# ИСПРАВЛЕННАЯ ЧАСТЬ users.py для квестов

# Добавляем в импорты в начале файла
from datetime import datetime, timedelta, timezone
from sqlalchemy.future import select
from app.models import User, Quest, UserQuestStatus
from app.schemas import QuestOut, StartQuestRequest, ClaimQuestRequest

# === ИСПРАВЛЕННЫЕ ЭНДПОИНТЫ ДЛЯ ДИНАМИЧЕСКИХ КВЕСТОВ ===

# ЗАМЕНИ ЭТОТ ЭНДПОИНТ В users.py

# ЗАМЕНИ ПОЛНОСТЬЮ ЭТОТ ЭНДПОИНТ В users.py

@router.get("/quests/dynamic", response_model=list[QuestOut])
async def list_dynamic_quests(telegram_id: int, db: AsyncSession = Depends(get_db)):
    """
    Список активных динамических квестов с полной информацией о статусе для пользователя
    """
    print(f"🔍 Загружаем квесты для пользователя {telegram_id}")
    
    # Получаем все активные квесты
    quests_res = await db.execute(select(Quest).where(Quest.active == True))
    quests = quests_res.scalars().all()
    print(f"📝 Найдено {len(quests)} активных квестов")
    
    # Получаем статусы пользователя для всех квестов
    statuses_res = await db.execute(
        select(UserQuestStatus).where(UserQuestStatus.user_id == telegram_id)
    )
    statuses = {s.quest_id: s for s in statuses_res.scalars().all()}
    print(f"📊 Статусы пользователя: {len(statuses)} записей")
    
    result = []
    now = datetime.now(timezone.utc)
    
    for quest in quests:
        status = statuses.get(quest.id)
        
        # Базовые данные квеста
        quest_data = {
            "id": quest.id,
            "title": quest.title,
            "quest_type": quest.quest_type,
            "url": quest.url,
            "reward_type": quest.reward_type,
            "reward_value": quest.reward_value,
            "description": quest.description,
            "active": quest.active,
            "timer_started_at": None,
            "completed": False,
            "reward_claimed": False,
            "can_claim": False,
            "seconds_left": None
        }
        
        if status:
            quest_data["completed"] = status.completed
            quest_data["reward_claimed"] = status.reward_claimed
            
            if status.timer_started_at:
                quest_data["timer_started_at"] = status.timer_started_at.isoformat()
                
                # Для YouTube квестов проверяем таймер
                if quest.quest_type == "youtube" and not status.reward_claimed:
                    elapsed = now - status.timer_started_at
                    wait_time = timedelta(minutes=10)
                    
                    print(f"⏰ Квест {quest.id}: прошло {elapsed.total_seconds()} секунд из {wait_time.total_seconds()}")
                    
                    if elapsed >= wait_time:
                        # Таймер истек - можно забирать награду
                        quest_data["can_claim"] = True
                        quest_data["completed"] = True
                        quest_data["seconds_left"] = 0
                        print(f"✅ Квест {quest.id}: можно забрать награду")
                    else:
                        # Таймер еще идет
                        remaining = wait_time - elapsed
                        quest_data["seconds_left"] = int(remaining.total_seconds())
                        quest_data["can_claim"] = False
                        quest_data["completed"] = False
                        print(f"⏳ Квест {quest.id}: осталось {quest_data['seconds_left']} секунд")
            
            # Для Telegram квестов проверяем подписку
            if quest.quest_type == "telegram" and not status.reward_claimed:
                try:
                    is_subscribed = await _check_tg_subscription(telegram_id, quest.url)
                    quest_data["completed"] = is_subscribed
                    quest_data["can_claim"] = is_subscribed
                    print(f"📱 Квест {quest.id}: подписка {'✅' if is_subscribed else '❌'}")
                except Exception as e:
                    print(f"❌ Ошибка проверки подписки для квеста {quest.id}: {e}")
                    quest_data["completed"] = False
                    quest_data["can_claim"] = False
        else:
            print(f"🆕 Квест {quest.id}: статус не найден (новый квест)")
        
        result.append(quest_data)
        print(f"📋 Квест {quest.id}: {quest_data}")
    
    print(f"🚀 Возвращаем {len(result)} квестов")
    return result


# ТАКЖЕ ЗАМЕНИ ЭНДПОИНТ /quests/start В users.py:

@router.post("/quests/start")
async def start_quest(body: StartQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    print(f"🚀 Запуск квеста {body.quest_id} для пользователя {telegram_id}")
    
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    res = await db.execute(select(Quest).where(Quest.id == body.quest_id, Quest.active == True))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found or inactive")

    print(f"📋 Квест найден: {quest.title} ({quest.quest_type})")

    # найдём/создадим статус
    res_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    us = res_us.scalar_one_or_none()
    if not us:
        print(f"🆕 Создаем новый статус квеста")
        us = UserQuestStatus(user_id=telegram_id, quest_id=quest.id)
        db.add(us)
    else:
        print(f"📊 Обновляем существующий статус квеста")

    # для youtube — фиксируем старт таймера
    if quest.quest_type == "youtube":
        us.timer_started_at = datetime.now(timezone.utc)
        us.completed = False  # сбрасываем при перезапуске
        us.reward_claimed = False
        print(f"⏰ YouTube таймер запущен: {us.timer_started_at}")

    await db.commit()
    print(f"✅ Квест успешно запущен")
    
    return {
        "status": "started", 
        "timer_started": us.timer_started_at.isoformat() if us.timer_started_at else None,
        "quest_type": quest.quest_type
    }
@router.post("/quests/claim")
async def claim_quest(body: ClaimQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    res = await db.execute(select(Quest).where(Quest.id == body.quest_id, Quest.active == True))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found or inactive")

    res_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    us = res_us.scalar_one_or_none()
    if not us:
        raise HTTPException(400, "Quest not started")

    if us.reward_claimed:
        return {"status": "already_claimed"}

    # проверяем условия
    if quest.quest_type == "youtube":
        if not us.timer_started_at:
            raise HTTPException(400, "Quest not started")
        if datetime.now(timezone.utc) < us.timer_started_at + timedelta(minutes=10):
            seconds_left = int((us.timer_started_at + timedelta(minutes=10) - datetime.now(timezone.utc)).total_seconds())
            raise HTTPException(400, f"Wait {seconds_left} seconds")

    elif quest.quest_type == "telegram":
        try:
            ok = await _check_tg_subscription(user_telegram_id=telegram_id, channel_ref=quest.url)
            if not ok:
                raise HTTPException(400, "Подпишитесь на канал и попробуйте снова")
        except Exception as e:
            print(f"Error checking telegram subscription: {e}")
            raise HTTPException(400, "Ошибка проверки подписки")

    # ИСПРАВЛЕННАЯ ЛОГИКА ВЫДАЧИ НАГРАДЫ
    now_utc = datetime.now(timezone.utc)
    
    # Проверяем активен ли буст у пользователя
    boost_active = False
    if user.boost_expiry:
        boost_expiry_utc = user.boost_expiry.astimezone(timezone.utc) if user.boost_expiry.tzinfo else user.boost_expiry.replace(tzinfo=timezone.utc)
        boost_active = boost_expiry_utc > now_utc
    
    # Выдаём награду
    if quest.reward_type == "boost":
        # Устанавливаем буст пользователю
        user.boost_multiplier = quest.reward_value
        user.boost_expiry = now_utc + timedelta(minutes=10)  # буст на 10 минут
        reward_message = f"Boost x{quest.reward_value} на 10 минут"
        
    elif quest.reward_type == "coins":
        reward = quest.reward_value
        # Если у пользователя активен буст, применяем его к награде
        if boost_active and user.boost_multiplier > 1:
            reward *= user.boost_multiplier
        user.coins += reward
        reward_message = f"{reward} coins"
        
    elif quest.reward_type == "energy":
        reward = quest.reward_value
        # Если у пользователя активен буст, применяем его к награде
        if boost_active and user.boost_multiplier > 1:
            reward *= user.boost_multiplier
        user.energy = min(user.max_energy, user.energy + reward)
        reward_message = f"{reward} energy"

    us.completed = True
    us.reward_claimed = True

    await db.commit()
    return {"status": "claimed", "reward": quest.reward_type, "amount": quest.reward_value, "message": reward_message}


# ИСПРАВЛЕННАЯ ЛОГИКА КЛИКОВ (заменить существующий эндпоинт)
@router.post("/{telegram_id}/click", response_model=ClickResponse)
async def click_coin(
    telegram_id: int,
    click_data: ClickRequest = ClickRequest(),
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        calculate_energy_regeneration(user)

        if user.energy < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough energy. Current: {user.energy}, Required: 10"
            )

        now = datetime.now(timezone.utc)
        
        # ИСПРАВЛЕННАЯ ПРОВЕРКА BOOST
        boost_active = False
        boost_multiplier = 1
        
        if user.boost_expiry:
            boost_expiry_utc = user.boost_expiry.astimezone(timezone.utc) if user.boost_expiry.tzinfo else user.boost_expiry.replace(tzinfo=timezone.utc)
            boost_active = boost_expiry_utc > now
            if boost_active:
                boost_multiplier = user.boost_multiplier

        coins_per_click = boost_multiplier  # используем boost_multiplier напрямую
        
        user.energy = max(user.energy - 10, 0)
        user.coins += coins_per_click
        user.total_clicks += 1

        old_level = user.level
        new_level = min(10, 1 + (user.total_clicks // 10000))
        if new_level != old_level:
            user.level = new_level
            user.max_energy = get_max_energy(new_level)
            user.energy = min(user.energy, user.max_energy)

        await db.commit()
        await db.refresh(user)

        return ClickResponse(
            coins=user.coins,
            energy=user.energy,
            total_clicks=user.total_clicks
        )

    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )