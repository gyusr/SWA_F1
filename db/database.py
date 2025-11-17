from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from fastapi import status
from fastapi.exceptions import HTTPException

from dotenv import load_dotenv
import os

load_dotenv()
DB_CONN = os.getenv("DB_CONN")

engine: AsyncEngine = None

async def context_get_conn():
    conn = None

    try:
        conn = await engine.connect()
        yield conn
    
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DB ERROR"
        )

    finally:
        if conn:
            await conn.close()
    