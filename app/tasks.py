from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.database import async_session_maker
from .models import User
from datetime import datetime, timedelta
from sqlalchemy import select


async def daily_farm():
    async with async_session_maker() as db:
        users = await db.execute(select(User))
        for user in users.scalars():
            if user.last_daily and (datetime.now() - user.last_daily).days < 1:
                continue
            user.coins += 500 * user.level
            user.last_daily = datetime.now()
        await db.commit()

def start_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(restore_energy, "interval", minutes=1)
    scheduler.add_job(daily_farm, "interval", hours=24)
    scheduler.start()
