from sqlalchemy import Column, Integer, String, DateTime, Boolean, BigInteger, Float, ForeignKey
from sqlalchemy.sql import func
from datetime import datetime
from app.database import Base
from sqlalchemy.orm import relationship 
from datetime import datetime, timedelta, timezone



class User(Base):
    __tablename__ = "users"

    telegram_id = Column(BigInteger, primary_key=True)
    username = Column(String(64), nullable=True)
    
    coins = Column(BigInteger, default=0, nullable=False)
    energy = Column(Integer, default=6500, nullable=False)
    max_energy = Column(Integer, default=6500, nullable=False)
    level = Column(Integer, default=1, nullable=False)
    total_clicks = Column(Integer, default=0, nullable=False)

    referral_code = Column(String(16), unique=True, nullable=False)
    referred_by = Column(String(16), nullable=True)

    # ВАЖНО: квесты «пригласить друзей» оставляем как у тебя было
    milestone_5_friends_claimed = Column(Boolean, default=False, nullable=False)
    reward_5_friends_claimed = Column(Boolean, default=False, nullable=False)
    reward_10_friends_claimed = Column(Boolean, default=False, nullable=False)

    # Бусты
    boost_expiry = Column(DateTime(timezone=True), nullable=True)
    boost_multiplier = Column(Integer, default=1, nullable=False)

    # Энергия / ежедневки
    last_energy_update = Column(DateTime(timezone=True),
                                server_default=func.now(),
                                nullable=False)
    daily_streak = Column(Integer, default=0, nullable=False)
    last_daily_login = Column(DateTime(timezone=True), nullable=True)

    # Роли
    role = Column(String(20), default="player", nullable=False)

    # Балансы валют
    tanki_blitz_balance = Column(Integer, default=0, nullable=False)
    mir_tankov_balance = Column(Integer, default=0, nullable=False)
    wot_blitz_balance = Column(Integer, default=0, nullable=False)

    # Лоченые коины
    locked_coins = Column(Integer, default=0, nullable=False)


class Quest(Base):
    __tablename__ = "quests"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    title        = Column(String(100), nullable=False)
    quest_type   = Column(String(20))  # "youtube" | "telegram" | (оставь другие свои типы при желании)
    reward_type  = Column(String(20))  # "coins" | "energy"
    reward_value = Column(Integer)
    active       = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.now)

    #  - youtube: ссылка на видео/канал
    #  - telegram: @username / https://t.me/username / numeric chat_id
    url          = Column(String(255), nullable=True)

    # опционально краткое описание
    description  = Column(String(255), nullable=True)
    boost_duration_minutes = Column(Integer, default=10, nullable=False)


class UserQuestStatus(Base):
    __tablename__ = "user_quest_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.telegram_id"), nullable=False)
    quest_id = Column(Integer, ForeignKey("quests.id"), nullable=False)

    completed = Column(Boolean, default=False, nullable=False)
    # для YouTube-таймера (10 мин). Для Telegram можно не трогать.
    timer_started_at = Column(DateTime, nullable=True)
    reward_claimed = Column(Boolean, default=False, nullable=False)

    user = relationship("User", backref="quest_statuses")
    quest = relationship("Quest", backref="user_statuses")


class PromoCode(Base):
    __tablename__ = "promocodes"

    code        = Column(String(32), primary_key=True)
    reward_type = Column(String(20))
    value       = Column(Integer)
    expiry      = Column(DateTime)
    created_at  = Column(DateTime, default=datetime.now)
    uses_left   = Column(Integer, default=1)


class ExchangeRate(Base):
    __tablename__ = "exchange_rates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_currency = Column(String(20), nullable=False)  
    to_currency = Column(String(20), nullable=False)    
    rate = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class ExchangeRequest(Base):
    __tablename__ = "exchange_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.telegram_id'), nullable=False)
    from_currency = Column(String(20), nullable=False)
    to_currency = Column(String(20), nullable=False)
    amount = Column(Integer, nullable=False)
    received_amount = Column(Integer, nullable=False)
    uid = Column(String(50), nullable=False)  
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.now)
    processed_at = Column(DateTime, nullable=True)
    
    user = relationship("User", backref="exchange_requests")


class UserPromoCode(Base):
    __tablename__ = "user_promocodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.telegram_id"), nullable=False)
    code = Column(String(32), ForeignKey("promocodes.code"), nullable=False)
    used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
