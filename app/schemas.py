from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime

# --- твои валюты ---
class CurrencyType(str, Enum):
    TANKI_BLITZ = "Танки Блитц"
    MIR_TANKOV = "Мир танков"
    WOT_BLITZ = "Wot Blitz"

# --- пользователи ---
class UserCreate(BaseModel):
    telegram_id: int = Field(..., gt=0)
    username: Optional[str] = Field(None, max_length=64)
    referred_by: Optional[str] = Field(None, max_length=16)
    role: Optional[str] = Field("player")

class UserOut(BaseModel):
    telegram_id: int
    username: Optional[str]
    coins: int
    level: int
    energy: int
    max_energy: int
    total_clicks: int
    referral_code: Optional[str]
    referred_by: Optional[str]
    tanki_blitz_balance: int = 0
    mir_tankov_balance: int = 0
    wot_blitz_balance: int = 0
    locked_coins: int = 0
    role: Optional[str] = Field("player")
    seconds_left: int = 0

    class Config:
        from_attributes = True

# --- старые квесты у тебя уже есть (если нужны эти схемы — оставь) ---
class QuestCreate(BaseModel):
    title: str
    quest_type: str
    reward_type: str
    reward_value: int

# --- обмен ---
class ExchangeRequestCreate(BaseModel):
    from_currency: str = "COIN"
    to_currency: CurrencyType
    amount: int
    uid: str

class SetExchangeRateRequest(BaseModel):
    currency: CurrencyType
    rate: float

class PromoCodeCreate(BaseModel):
    code: Optional[str] = None
    reward_type: str
    value: int
    uses_left: int = 1

class QuestStatusResponse(BaseModel):
    """Ответ для статуса квеста"""
    quest_id: int
    quest_type: str
    title: str
    completed: bool
    reward_claimed: bool
    
    # YouTube specific fields
    timer_started_at: Optional[str] = None
    seconds_left: int = 0
    can_claim: bool = False
    
    # Telegram specific fields  
    can_subscribe: bool = False

class YouTubeQuestStartResponse(BaseModel):
    """Ответ при начале YouTube квеста"""
    status: str  # "timer_started"
    quest_id: int
    timer_started_at: str
    seconds_left: int
    can_claim: bool
    youtube_url: Optional[str] = None

class QuestClaimResponse(BaseModel):
    """Ответ при получении награды"""
    status: str  # "claimed"
    quest_id: int
    reward_type: str
    reward_value: int
    user_coins: int
    user_energy: Optional[int] = None

class TelegramQuestResponse(BaseModel):
    """Ответ для Telegram квеста"""
    status: str  # "not_subscribed", "completed", "already_completed"
    message: str
    quest_id: Optional[int] = None
    reward_type: Optional[str] = None
    reward_value: Optional[int] = None
    user_coins: Optional[int] = None
    user_energy: Optional[int] = None
    telegram_url: Optional[str] = None
    need_subscription: bool = False

class DynamicQuestResponse(BaseModel):
    """Ответ для списка динамических квестов"""
    id: int
    title: str
    description: Optional[str] = None
    quest_type: str
    reward_type: str
    reward_value: int
    url: Optional[str] = None
    completed: bool
    reward_claimed: bool
    quest_status: str  # "not_started", "timer_running", "ready_to_claim", "completed", "not_completed"
    
    # YouTube specific
    timer_started_at: Optional[str] = None
    seconds_left: int = 0
    can_claim: bool = False
    
    # Telegram specific
    can_subscribe: bool = False

# ===== СХЕМЫ ДЛЯ ADMIN.PY (НЕДОСТАЮЩИЕ) =====

class QuestUpsert(BaseModel):
    """Схема для создания/обновления квеста"""
    title: str
    quest_type: str  # "youtube" | "telegram"
    reward_type: str  # "coins" | "energy" | "boost"
    reward_value: int
    url: Optional[str] = None
    description: Optional[str] = None
    active: bool = True

class QuestOut(BaseModel):
    """Схема для вывода квеста"""
    id: int
    title: str
    quest_type: str
    reward_type: str
    reward_value: int
    url: Optional[str] = None
    description: Optional[str] = None
    active: bool
    created_at: Optional[datetime] = None
    boost_duration_minutes: int = 10

    class Config:
        from_attributes = True

# ===== СХЕМЫ ДЛЯ USERS.PY (НЕДОСТАЮЩИЕ) =====

class StartQuestRequest(BaseModel):
    """Запрос на начало квеста"""
    quest_id: int

class ClaimQuestRequest(BaseModel):
    """Запрос на получение награды"""
    quest_id: int