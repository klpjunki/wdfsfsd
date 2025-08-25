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
import re
from app.schemas import QuestOut





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



YOUTUBE_WAIT = timedelta(minutes=10)


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




QUEST_DURATION = 600  # 10 минут для YouTube квестов

def _normalize_channel_ref(raw: str) -> str:
    """Нормализует ссылку на Telegram канал"""
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
    """Проверяет подписку пользователя на Telegram канал"""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(500, "TELEGRAM_BOT_TOKEN not set")
    
    chat_id = _normalize_channel_ref(channel_ref)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChatMember"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params={"chat_id": chat_id, "user_id": user_telegram_id})
            data = r.json()
            
            if not data.get("ok"):
                return False
                
            status = data["result"]["status"]
            return status in ("member", "administrator", "creator")
    except Exception as e:
        print(f"Error checking Telegram subscription: {e}")
        return False

# ===== YOUTUBE КВЕСТЫ =====

@router.post("/quests/youtube/{quest_id}/start")
async def start_youtube_quest(
    quest_id: int, 
    telegram_id: int, 
    db: AsyncSession = Depends(get_db)
):
    """Этап 1: Начать YouTube квест (запуск таймера)"""
    
    # Проверяем пользователя
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    # Проверяем квест
    result = await db.execute(
        select(Quest).where(
            Quest.id == quest_id,
            Quest.active == True,
            Quest.quest_type == "youtube"
        )
    )
    quest = result.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "YouTube quest not found or inactive")
    
    # Получаем или создаем статус квеста
    result_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    user_status = result_us.scalar_one_or_none()
    
    if not user_status:
        user_status = UserQuestStatus(user_id=telegram_id, quest_id=quest.id)
        db.add(user_status)
        await db.flush()
    
    # Проверяем, не завершен ли уже квест
    if user_status.reward_claimed:
        raise HTTPException(400, "Quest already completed")
    
    # Запускаем таймер (если еще не запущен)
    if not user_status.timer_started_at:
        user_status.timer_started_at = datetime.now(timezone.utc)
        user_status.completed = False
        user_status.reward_claimed = False
        await db.commit()
    
    # Вычисляем оставшееся время
    elapsed = (datetime.now(timezone.utc) - user_status.timer_started_at).total_seconds()
    seconds_left = max(0, int(QUEST_DURATION - elapsed))
    
    return {
        "status": "timer_started",
        "quest_id": quest.id,
        "timer_started_at": user_status.timer_started_at.isoformat(),
        "seconds_left": seconds_left,
        "can_claim": seconds_left == 0,
        "youtube_url": quest.url
    }

@router.post("/quests/youtube/{quest_id}/claim")
async def claim_youtube_quest(
    quest_id: int, 
    telegram_id: int, 
    db: AsyncSession = Depends(get_db)
):
    """Этап 3: Забрать награду за YouTube квест"""
    
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    # Проверяем квест
    result = await db.execute(
        select(Quest).where(
            Quest.id == quest_id,
            Quest.active == True,
            Quest.quest_type == "youtube"
        )
    )
    quest = result.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "YouTube quest not found")
    
    # Получаем статус квеста
    result_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    user_status = result_us.scalar_one_or_none()
    
    if not user_status or not user_status.timer_started_at:
        raise HTTPException(400, "Quest not started. Start the quest first.")
    
    if user_status.reward_claimed:
        raise HTTPException(400, "Reward already claimed")
    
    # Проверяем, закончился ли таймер
    now = datetime.now(timezone.utc)
    elapsed = (now - user_status.timer_started_at).total_seconds()
    seconds_left = max(0, int(QUEST_DURATION - elapsed))
    
    if seconds_left > 0:
        raise HTTPException(
            400, 
            f"Timer not finished. Wait {seconds_left} more seconds."
        )
    
    # Выдаем награду
    user_status.completed = True
    user_status.reward_claimed = True
    
    if quest.reward_type == "coins":
        user.coins = (user.coins or 0) + int(quest.reward_value or 0)
    elif quest.reward_type == "energy":
        max_energy = getattr(user, "max_energy", None)
        current_energy = (getattr(user, "energy", None) or 0)
        if max_energy is not None:
            user.energy = min(max_energy, current_energy + int(quest.reward_value or 0))
        else:
            user.energy = current_energy + int(quest.reward_value or 0)
    elif quest.reward_type == "boost":
        boost_duration = getattr(quest, "boost_duration_minutes", 10)
        user.boost_multiplier = max(getattr(user, "boost_multiplier", 1), 2)
        user.boost_expiry = now + timedelta(minutes=boost_duration)
    
    await db.commit()
    await db.refresh(user)
    await db.refresh(user_status)
    
    return {
        "status": "claimed",
        "quest_id": quest.id,
        "reward_type": quest.reward_type,
        "reward_value": quest.reward_value,
        "user_coins": user.coins,
        "user_energy": getattr(user, "energy", None)
    }

@router.get("/quests/youtube/{quest_id}/status")
async def youtube_quest_status(
    quest_id: int,
    telegram_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Этап 2: Проверка статуса YouTube квеста (например, при перезагрузке фронта)"""
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")

    result = await db.execute(
        select(Quest).where(
            Quest.id == quest_id,
            Quest.active == True,
            Quest.quest_type == "youtube"
        )
    )
    quest = result.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "YouTube quest not found")

    # Проверяем статус
    result_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    user_status = result_us.scalar_one_or_none()
    if not user_status or not user_status.timer_started_at:
        return {
            "status": "not_started",
            "quest_id": quest.id,
            "seconds_left": None,
            "can_claim": False,
            "youtube_url": quest.url
        }

    # Считаем сколько осталось
    elapsed = (datetime.now(timezone.utc) - user_status.timer_started_at).total_seconds()
    seconds_left = max(0, int(QUEST_DURATION - elapsed))

    return {
        "status": "in_progress" if not user_status.reward_claimed else "completed",
        "quest_id": quest.id,
        "seconds_left": seconds_left,
        "can_claim": seconds_left == 0 and not user_status.reward_claimed,
        "youtube_url": quest.url
    }

# ===== TELEGRAM КВЕСТЫ =====

@router.post("/quests/telegram/{quest_id}/subscribe")
async def subscribe_telegram_quest(
    quest_id: int, 
    telegram_id: int, 
    db: AsyncSession = Depends(get_db)
):
    """Подписаться на Telegram канал и получить награду"""
    
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    # Проверяем квест
    result = await db.execute(
        select(Quest).where(
            Quest.id == quest_id,
            Quest.active == True,
            Quest.quest_type == "telegram"
        )
    )
    quest = result.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Telegram quest not found or inactive")
    
    # Получаем или создаем статус квеста
    result_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    user_status = result_us.scalar_one_or_none()
    
    if not user_status:
        user_status = UserQuestStatus(user_id=telegram_id, quest_id=quest.id)
        db.add(user_status)
        await db.flush()
    
    if user_status.reward_claimed:
        return {
            "status": "already_completed",
            "message": "Вы уже выполнили это задание"
        }
    
    # Проверяем подписку
    is_subscribed = await _check_tg_subscription(telegram_id, quest.url)
    
    if not is_subscribed:
        return {
            "status": "not_subscribed",
            "message": "Подпишитесь на канал и попробуйте снова",
            "telegram_url": quest.url,
            "need_subscription": True
        }
    
    # Выдаем награду сразу при успешной проверке
    user_status.completed = True
    user_status.reward_claimed = True
    
    if quest.reward_type == "coins":
        user.coins = (user.coins or 0) + int(quest.reward_value or 0)
    elif quest.reward_type == "energy":
        max_energy = getattr(user, "max_energy", None)
        current_energy = (getattr(user, "energy", None) or 0)
        if max_energy is not None:
            user.energy = min(max_energy, current_energy + int(quest.reward_value or 0))
        else:
            user.energy = current_energy + int(quest.reward_value or 0)
    elif quest.reward_type == "boost":
        boost_duration = getattr(quest, "boost_duration_minutes", 10)
        user.boost_multiplier = max(getattr(user, "boost_multiplier", 1), 2)
        user.boost_expiry = datetime.now(timezone.utc) + timedelta(minutes=boost_duration)
    
    await db.commit()
    await db.refresh(user)
    await db.refresh(user_status)
    
    return {
        "status": "completed",
        "message": "Задание выполнено! Награда получена.",
        "quest_id": quest.id,
        "reward_type": quest.reward_type,
        "reward_value": quest.reward_value,
        "user_coins": user.coins,
        "user_energy": getattr(user, "energy", None)
    }

# ===== ОБНОВЛЕННЫЙ СПИСОК КВЕСТОВ =====

@router.get("/quests/dynamic")
async def get_dynamic_quests(telegram_id: int, db: AsyncSession = Depends(get_db)):
    """Получить список всех квестов с их статусами"""
    
    now = datetime.now(timezone.utc)
    
    # Получаем все активные квесты
    result_q = await db.execute(select(Quest).where(Quest.active == True))
    quests = result_q.scalars().all()
    
    # Получаем статусы пользователя
    result_us = await db.execute(
        select(UserQuestStatus).where(UserQuestStatus.user_id == telegram_id)
    )
    statuses = result_us.scalars().all()
    status_map = {s.quest_id: s for s in statuses}
    
    quest_list = []
    
    for quest in quests:
        user_status = status_map.get(quest.id)
        
        quest_data = {
            "id": quest.id,
            "title": quest.title,
            "description": getattr(quest, "description", None),
            "quest_type": quest.quest_type,
            "reward_type": getattr(quest, "reward_type", None),
            "reward_value": getattr(quest, "reward_value", None),
            "url": getattr(quest, "url", None),
            "completed": False,
            "reward_claimed": False
        }
        
        if quest.quest_type == "youtube":
            # YouTube квест логика
            timer_started = None
            seconds_left = 0
            can_claim = False
            quest_status = "not_started"  # not_started, timer_running, ready_to_claim, completed
            
            if user_status:
                quest_data["completed"] = bool(user_status.completed)
                quest_data["reward_claimed"] = bool(user_status.reward_claimed)
                
                if user_status.reward_claimed:
                    quest_status = "completed"
                elif user_status.timer_started_at:
                    timer_started = user_status.timer_started_at
                    elapsed = (now - user_status.timer_started_at).total_seconds()
                    seconds_left = max(0, int(QUEST_DURATION - elapsed))
                    
                    if seconds_left > 0:
                        quest_status = "timer_running"
                    else:
                        quest_status = "ready_to_claim"
                        can_claim = True
            
            quest_data.update({
                "quest_status": quest_status,
                "timer_started_at": timer_started.isoformat() if timer_started else None,
                "seconds_left": seconds_left,
                "can_claim": can_claim
            })
            
        elif quest.quest_type == "telegram":
            # Telegram квест логика
            quest_status = "not_completed"  # not_completed, completed
            
            if user_status and user_status.reward_claimed:
                quest_status = "completed"
                quest_data["completed"] = True
                quest_data["reward_claimed"] = True
            
            quest_data.update({
                "quest_status": quest_status,
                "can_subscribe": quest_status != "completed"
            })
        
        quest_list.append(quest_data)
    
    return quest_list

# ===== ПРОВЕРКА СТАТУСА КВЕСТА =====

@router.get("/quests/{quest_id}/status")
async def get_quest_status(
    quest_id: int, 
    telegram_id: int, 
    db: AsyncSession = Depends(get_db)
):
    """Получить детальный статус конкретного квеста"""
    
    # Проверяем квест
    result = await db.execute(
        select(Quest).where(Quest.id == quest_id, Quest.active == True)
    )
    quest = result.scalar_one_or_none()
    if not quest:
        raise HTTPException(404, "Quest not found")
    
    # Получаем статус пользователя
    result_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    user_status = result_us.scalar_one_or_none()
    
    response = {
        "quest_id": quest.id,
        "quest_type": quest.quest_type,
        "title": quest.title,
        "completed": False,
        "reward_claimed": False
    }
    
    if quest.quest_type == "youtube":
        if user_status and user_status.timer_started_at:
            now = datetime.now(timezone.utc)
            elapsed = (now - user_status.timer_started_at).total_seconds()
            seconds_left = max(0, int(QUEST_DURATION - elapsed))
            
            response.update({
                "timer_started_at": user_status.timer_started_at.isoformat(),
                "seconds_left": seconds_left,
                "can_claim": seconds_left == 0 and not user_status.reward_claimed,
                "completed": bool(user_status.completed),
                "reward_claimed": bool(user_status.reward_claimed)
            })
    
    elif quest.quest_type == "telegram":
        if user_status:
            response.update({
                "completed": bool(user_status.completed),
                "reward_claimed": bool(user_status.reward_claimed)
            })
    
    return response