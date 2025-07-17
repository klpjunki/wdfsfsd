import random
import string
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Quest, PromoCode  
from app.database import async_session_maker  
from app.schemas import UserCreate 

async def get_user(db: AsyncSession, telegram_id: int):
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    return result.scalar_one_or_none()

async def get_user_by_referral_code(db: AsyncSession, code: str):
    result = await db.execute(select(User).where(User.referral_code == code))
    return result.scalar_one_or_none()

async def generate_referral_code(db: AsyncSession):  
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(chars, k=8))
        existing_user = await get_user_by_referral_code(db, code)  
        if not existing_user:
            return code

async def create_user(db: AsyncSession, user_data: UserCreate):
    referral_code = await generate_referral_code(db)  
    user = User(
        telegram_id=user_data.telegram_id,
        username=user_data.username,
        referral_code=referral_code,
        referred_by=user_data.referred_by,
        max_energy=6500
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

