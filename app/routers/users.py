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



QUEST_DURATION = 600  # 10 минут

# ----------------- START QUEST -----------------
@router.post("/quests/start")
async def start_quest(body: StartQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    print(f"[start] start_quest request: telegram_id={telegram_id}, quest_id={body.quest_id}")

    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    res = await db.execute(select(Quest).where(Quest.id == body.quest_id, Quest.active == True))
    quest = res.scalar_one_or_none()
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found or inactive")

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
        await db.flush()

    # Запускаем таймер только если еще не был запущен
    if not us.timer_started_at:
        now = datetime.now(timezone.utc)
        us.timer_started_at = now
        us.completed = False
        us.reward_claimed = False
        await db.commit()
        await db.refresh(us)
        print(f"[start] timer_started_at set to {us.timer_started_at.isoformat()}")
    else:
        print(f"[start] timer already started at {us.timer_started_at.isoformat()}")

    seconds_left = 0
    if us.timer_started_at:
        elapsed = (datetime.now(timezone.utc) - us.timer_started_at).total_seconds()
        seconds_left = max(0, int(QUEST_DURATION - elapsed))

    return {
        "status": "started",
        "timer_started_at": us.timer_started_at.isoformat() if us.timer_started_at else None,
        "quest_type": quest.quest_type,
        "seconds_left": seconds_left
    }

# ----------------- GET DYNAMIC QUESTS -----------------
@router.get("/quests/dynamic")
async def get_dynamic_quests(telegram_id: int, db: AsyncSession = Depends(get_db)):
    print(f"[dynamic] get_dynamic_quests for telegram_id={telegram_id}")
    now = datetime.now(timezone.utc)

    res_q = await db.execute(select(Quest).where(Quest.active == True))
    quests = res_q.scalars().all()

    res_us = await db.execute(select(UserQuestStatus).where(UserQuestStatus.user_id == telegram_id))
    statuses = res_us.scalars().all()
    status_map = {s.quest_id: s for s in statuses}

    out = []
    for q in quests:
        us = status_map.get(q.id)
        timer_started = None
        seconds_left = 0
        can_claim = False
        reward_claimed = False
        completed = False

        if us:
            timer_started = us.timer_started_at
            reward_claimed = bool(us.reward_claimed)
            completed = bool(us.completed)
            if us.timer_started_at:
                elapsed = (now - us.timer_started_at).total_seconds()
                seconds_left = max(0, int(QUEST_DURATION - elapsed))
                if seconds_left == 0 and not us.reward_claimed:
                    can_claim = True

        out.append({
            "id": q.id,
            "title": q.title,
            "description": getattr(q, "description", None),
            "quest_type": q.quest_type,
            "timer_started_at": timer_started.isoformat() if timer_started else None,
            "seconds_left": seconds_left,
            "can_claim": can_claim,
            "reward_claimed": reward_claimed,
            "completed": completed,
            "reward_type": getattr(q, "reward_type", None),
            "reward_value": getattr(q, "reward_value", None),
            "url": getattr(q, "url", None)
        })

    return out

# ----------------- CLAIM QUEST -----------------
@router.post("/quests/claim")
async def claim_quest(body: ClaimQuestRequest, telegram_id: int, db: AsyncSession = Depends(get_db)):
    print(f"[claim] claim_quest: telegram_id={telegram_id}, quest_id={body.quest_id}")
    quest_id = body.quest_id

    user = await db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    res_q = await db.execute(select(Quest).where(Quest.id == quest_id, Quest.active == True))
    quest = res_q.scalar_one_or_none()
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found or inactive")

    res_us = await db.execute(
        select(UserQuestStatus).where(
            UserQuestStatus.user_id == telegram_id,
            UserQuestStatus.quest_id == quest.id
        )
    )
    us = res_us.scalar_one_or_none()
    if not us or not us.timer_started_at:
        raise HTTPException(status_code=400, detail="Quest not started")

    if us.reward_claimed:
        raise HTTPException(status_code=400, detail="Reward already claimed")

    now = datetime.now(timezone.utc)
    elapsed = (now - us.timer_started_at).total_seconds()
    seconds_left = max(0, int(QUEST_DURATION - elapsed))
    if seconds_left > 0:
        raise HTTPException(status_code=400, detail=f"Timer not finished, seconds_left: {seconds_left}")

    # Выдача награды
    us.reward_claimed = True
    us.completed = True

    rt = getattr(quest, "reward_type", None)
    rv = getattr(quest, "reward_value", 0)

    if rt == "coins":
        user.coins = (user.coins or 0) + int(rv or 0)
    elif rt == "energy":
        max_e = getattr(user, "max_energy", None)
        cur = (getattr(user, "energy", None) or 0)
        if max_e is not None:
            user.energy = min(max_e, cur + int(rv or 0))
        else:
            user.energy = cur + int(rv or 0)
    elif rt == "boost":
        dur = getattr(quest, "boost_duration_minutes", 10)
        user.boost_multiplier = max(getattr(user, "boost_multiplier", 1), 2)
        user.boost_expiry = now + timedelta(minutes=dur)
    else:
        print(f"[claim] unknown reward_type: {rt}")

    db.add(user)
    await db.commit()
    await db.refresh(user)
    await db.refresh(us)

    return {
        "status": "claimed",
        "quest_id": quest.id,
        "reward_type": rt,
        "reward_value": rv,
        "user_coins": user.coins,
        "user_energy": getattr(user, "energy", None)
    }
