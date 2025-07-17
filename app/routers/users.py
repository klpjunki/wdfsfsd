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
from app.services.telegram_checker import check_telegram_subscription
from app.models import ExchangeRate
from app.schemas import ExchangeRequestCreate, SetExchangeRateRequest, CurrencyType
from sqlalchemy import delete
import os  
from sqlalchemy.orm import Session

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
    youtube_subscribed: bool = False
    youtube_reward_claimed: bool = False
    youtube_timer_started: Optional[datetime] = None
    telegram_subscribed: bool = False
    telegram_reward_claimed: bool = False
    telegram_timer_started: Optional[datetime] = None
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
        'reward_10_friends_claimed', 'youtube_subscribed',
        'telegram_subscribed', mode='before'
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
GLOBAL_MARKET_TIMER_END = GLOBAL_MARKET_TIMER_START + timedelta(days=90)


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
        "role": user.role,
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
        youtube_timer_started=None,
        telegram_timer_started=None,
        role=user_data.role or "player",
        last_energy_update=datetime.now(timezone.utc)
                )

    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
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
        "youtube_subscribed": user.youtube_subscribed,
        "youtube_reward_claimed": user.youtube_reward_claimed,
        "youtube_timer_started": user.youtube_timer_started,
        "telegram_subscribed": user.telegram_subscribed,
        "telegram_reward_claimed": user.telegram_reward_claimed,
        "telegram_timer_started": user.telegram_timer_started,
        "daily_streak": user.daily_streak,
        "last_daily_login": user.last_daily_login,
        "role": user.role,
        "boost_expiry": user.boost_expiry,
        "boost_multiplier": user.boost_multiplier,
        "boost_active": user.boost_active,
        "seconds_left": seconds_left
    }

@router.get("/{telegram_id}/referral_progress")
async def get_referral_progress(
    telegram_id: int, 
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    result = await db.execute(
        select(User).where(User.referred_by == user.referral_code)
    )
    referrals = result.scalars().all()
    referral_count = len(referrals)
    quest_5_status = "completed" if referral_count >= 5 else "in_progress"
    quest_10_status = "completed" if referral_count >= 10 else "in_progress"
    reward_5_claimed = user.reward_5_friends_claimed
    reward_10_claimed = user.reward_10_friends_claimed
    youtube_completed = user.youtube_subscribed
    youtube_reward_claimed = user.youtube_reward_claimed
    youtube_timer_started = user.youtube_timer_started

    youtube_can_claim = False
    youtube_time_left = None

    if youtube_completed and not youtube_reward_claimed and youtube_timer_started:
        now = datetime.now(timezone.utc)
        time_elapsed = (now - youtube_timer_started.replace(tzinfo=timezone.utc)).total_seconds()
        if time_elapsed >= 600:  
            youtube_can_claim = True
            youtube_time_left = 0
        else:
            youtube_can_claim = False
            youtube_time_left = int(600 - time_elapsed)
    elif youtube_completed and not youtube_reward_claimed:
        youtube_can_claim = False
        youtube_time_left = None
    telegram_subscribed = check_telegram_subscription(telegram_id)
    
    return {
        "referral_count": referral_count,
        "quest_5_friends": {
            "status": quest_5_status,
            "reward_claimed": reward_5_claimed,
            "can_claim": referral_count >= 5 and not reward_5_claimed
        },
        "quest_10_friends": {
            "status": quest_10_status,
            "reward_claimed": reward_10_claimed,
            "can_claim": referral_count >= 10 and not reward_10_claimed
        },
        "youtube_quest": {
            "completed": youtube_completed,
            "reward_claimed": youtube_reward_claimed,
            "can_claim": youtube_can_claim,
            "time_left": youtube_time_left,
            "timer_started": youtube_timer_started.isoformat() if youtube_timer_started else None
        },
        "telegram_quest": {
            "completed": user.telegram_subscribed,
            "reward_claimed": user.telegram_reward_claimed,
            "can_claim": telegram_subscribed and not user.telegram_reward_claimed,
            "time_left": None  
        }
    }
@router.post("/{telegram_id}/youtube_subscribe")
async def youtube_subscribe(telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.youtube_subscribed:
        raise HTTPException(400, "YouTube task already completed")
    user.youtube_subscribed = True
    user.youtube_timer_started = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return {"status": "success", "message": "YouTube timer started"}

@router.post("/{telegram_id}/claim_youtube_reward")
async def claim_youtube_reward(
    telegram_id: int,
    db: AsyncSession = Depends(get_db)
):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.youtube_subscribed:
        raise HTTPException(400, "Complete YouTube task first")
    if user.youtube_reward_claimed:
        raise HTTPException(400, "Reward already claimed")
    if not user.youtube_timer_started:
        raise HTTPException(400, "Timer not started")
    now = datetime.now(timezone.utc)
    elapsed = (now - user.youtube_timer_started.replace(tzinfo=timezone.utc)).total_seconds()
    
    if elapsed < 60:  
        left = 60 - elapsed
        m, s = divmod(int(left), 60)
        raise HTTPException(400, f"Wait {m}:{s:02d} before claiming reward")
    user.boost_expiry = now + timedelta(minutes=10)
    user.boost_multiplier = 10
    user.youtube_reward_claimed = True
    
    await db.commit()
    await db.refresh(user)
    
    return {
        "status": "success",
        "boost_expiry": user.boost_expiry.isoformat(),
        "boost_multiplier": user.boost_multiplier
    }
@router.post("/{telegram_id}/telegram_subscribe")
async def telegram_subscribe(telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.telegram_subscribed:
        raise HTTPException(400, "Telegram task already completed")
    user.telegram_subscribed = True
    user.telegram_timer_started = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return {"status": "success", "message": "Telegram timer started"}

async def get_user_by_telegram_id(db: AsyncSession, telegram_id: int):
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    return result.scalars().first()

@router.post("/{telegram_id}/claim_telegram_reward")
async def claim_telegram_reward(telegram_id: int, db: AsyncSession = Depends(get_db)):
    telegram_subscribed = check_telegram_subscription(telegram_id)
    if not telegram_subscribed:
        raise HTTPException(400, "Подпишитесь на канал для получения награды!")

    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        raise HTTPException(404, "Пользователь не найден")

    if user.telegram_reward_claimed:
        raise HTTPException(400, "Награда уже получена")
    user.boost_active = True
    user.boost_multiplier = 10  
    user.boost_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    user.telegram_reward_claimed = True

    await db.commit()

    return {
        "message": "Буст активирован: +10 за клик на 10 минут!",
        "boost_active": user.boost_active,
        "boost_multiplier": user.boost_multiplier,
        "boost_expiry": user.boost_expiry.isoformat()
    }
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
    if promocode.reward_type == "coins":
        user.coins += promocode.value
    elif promocode.reward_type == "energy":
       user.energy = user.energy + promocode.value
    promocode.uses_left -= 1
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


@router.get("/exchange-rate")
async def get_exchange_rate(to_currency: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ExchangeRate.rate)
        .where(ExchangeRate.to_currency == to_currency)
        .order_by(ExchangeRate.last_updated.desc())
        .limit(1)
    )
    rate = result.scalar_one_or_none()
    if rate is None:
        return {"rate": 1}
    return {"rate": rate}
