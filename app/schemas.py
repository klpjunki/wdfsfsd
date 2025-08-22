from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

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

# ============== ИСПРАВЛЕННЫЕ СХЕМЫ ДЛЯ ДИНАМИЧЕСКИХ КВЕСТОВ ==============

# админ создаёт/обновляет квест - ДОБАВЛЕН "boost"
class QuestUpsert(BaseModel):
    title: str
    quest_type: Literal["youtube", "telegram"]
    url: str
    reward_type: Literal["coins", "energy", "boost"]  # ДОБАВЛЕН "boost"
    reward_value: int = Field(ge=1)
    description: Optional[str] = None
    active: bool = True

# выдаём квест на фронт - ДОБАВЛЕН "boost"
class QuestOut(BaseModel):
    id: int
    title: str
    quest_type: Literal["youtube", "telegram"]
    url: Optional[str] = None
    reward_type: Literal["coins", "energy", "boost"]  # ДОБАВЬ "boost" если его нет
    reward_value: int
    description: Optional[str] = None
    active: Optional[bool] = True
    
    timer_started_at: Optional[str] = None
    completed: bool = False
    reward_claimed: bool = False
    can_claim: bool = False
    seconds_left: Optional[int] = None

    class Config:
        from_attributes = Truetes = True

# пользователь: начало выполнения и получение награды
class StartQuestRequest(BaseModel):
    quest_id: int

class ClaimQuestRequest(BaseModel):
    quest_id: int