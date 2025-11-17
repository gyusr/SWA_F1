from fastapi import FastAPI
from db import database
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Server...")
    
    # DB 엔진 생성
    database.engine = create_async_engine(
        database.DB_CONN,
        pool_size=10,
        max_overflow=0,
        pool_recycle=300
    )
    print(f"create database engine complete")

    yield # 기준점

    print("Shutting Server...")

    # DB 엔진 종료
    await database.engine.dispose()
    print("dispose database engine complete")