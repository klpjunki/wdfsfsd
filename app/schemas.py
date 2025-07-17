from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class CurrencyType(str, Enum):
    TANKI_BLITZ = "Танки Блитц"
    MIR_TANKOV = "Мир танков"
    WOT_BLITZ = "Wot Blitz"

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

class QuestCreate(BaseModel):
    title: str
    quest_type: str
    reward_type: str
    reward_value: int

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
