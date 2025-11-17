from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as  StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os

from utils import common
from utils import middleware


# FastAPI APP 생성
app = FastAPI(
    lifespan=common.lifespan
)


# Template Engine 생성 및 StaticFiles
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")


# Middleware 설정
app.add_middleware(middleware.MethodOverrideMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    max_age=-1
)
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)


# APIRouter 설정
from routes import meetings, user
app.include_router(meetings.router)
app.include_router(user.router)


# Exception Handler 설정
from utils import exc_handler
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
app.add_exception_handler(RequestValidationError, exc_handler.custom_request_validation_error_handler)


# Main Page 이동
@app.get("/")
async def main_page():
    return RedirectResponse(
        url="/user/login",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT
    )
