from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.database import engine, Base
from app.routers import users, quests, admin
from app.scheduler import start_scheduler, stop_scheduler
import logging
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import os


from dotenv import load_dotenv

load_dotenv()
print("ADMIN_SECRET =", os.getenv("ADMIN_SECRET"))  


logger = logging.getLogger(__name__)            
logging.basicConfig(level=logging.INFO)         

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tank-coin.ru",
        "https://*.telegram.org"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    import traceback
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

app.include_router(users.router, prefix="/api/users")
app.include_router(admin.router, prefix="/api/admin")

@app.get("/")
async def root():
    return {"message": "Gacha backend works!"}

@app.on_event("startup")
async def startup_event():
    for route in app.routes:
        print(f"{route.path} - {route.methods}")

from fastapi import Request

@app.get("/debug-headers")
async def debug_headers(request: Request):
    return dict(request.headers)