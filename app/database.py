import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "gacha_coin")
DATABASE_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_async_engine(
    DATABASE_URL,
    echo=True,               
    pool_size=10,            
    max_overflow=20,         
    pool_recycle=1800,       
    pool_pre_ping=True,      
    pool_timeout=30,         
    connect_args={
        "connect_timeout": 10  
    }
)

async_session_maker = sessionmaker(
    engine, 
    class_=AsyncSession,
    expire_on_commit=False  
)
Base = declarative_base()

async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
