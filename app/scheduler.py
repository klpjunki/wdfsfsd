from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select, update
from app.database import async_session_maker
from app.models import User
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def get_max_energy(level: int) -> int:
    return 6500 + 500 * (min(level, 10) - 1)

async def regenerate_energy_for_all_users():
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            now = datetime.now()
            updated_users = []
            for user in users:
                if not user.last_energy_update:
                    user.last_energy_update = now
                    continue
                time_diff = (now - user.last_energy_update).total_seconds()
                energy_to_restore = int(time_diff / 6)
                if energy_to_restore > 0:
                    base_max_energy = get_max_energy(user.level)
                    milestone_claimed = bool(getattr(user, 'milestone_5_friends_claimed', False))
                    if user.energy < base_max_energy:
                        new_energy = min(user.energy + energy_to_restore, base_max_energy)
                        await session.execute(
                            update(User)
                            .where(User.telegram_id == user.telegram_id)
                            .values(
                                energy=new_energy,
                                last_energy_update=now
                            )
                        )
                        updated_users.append(user.telegram_id)
            await session.commit()
            if updated_users:
                logger.info(f"Energy regenerated for {len(updated_users)} users")
    except Exception as e:
        logger.error(f"Error in energy regeneration: {str(e)}")

async def auto_daily_bonus_for_all_users():
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            now = datetime.now(timezone.utc)
            for user in users:
                last_login = user.last_daily_login
                if last_login:
                    days_since_last = (now.date() - last_login.date()).days
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
            await session.commit()
            logger.info("Бонусы выданы автоматически всем пользователям")
    except Exception as e:
        logger.error(f"Ошибка при выдаче бонусов: {str(e)}")

scheduler = None

def start_scheduler():
    global scheduler
    try:
        scheduler = AsyncIOScheduler(timezone='Europe/Moscow')
        scheduler.add_job(
            regenerate_energy_for_all_users,
            'interval',
            minutes=1,
            id='energy_regeneration',
            replace_existing=True
        )
        scheduler.add_job(
            auto_daily_bonus_for_all_users,
            'cron',
            hour=0,
            minute=0,
            id='daily_bonus',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Scheduler started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        return False

def stop_scheduler():
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
