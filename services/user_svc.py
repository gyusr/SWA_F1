from pydantic import EmailStr
from schemas import user_schemas
from fastapi import status, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic import ValidationError
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError


def correct_user_form(
    name: str = None,
    email: EmailStr = None,
    password: str = None
):
    try:
        user = user_schemas.CorrectUserForm(
            name=name,
            email=email,
            password=password
        )

        return user

    except ValidationError as e:
        print(e.errors())
        raise RequestValidationError(e.errors())
    

async def get_user_by_email(
    conn: Connection,
    email: EmailStr
):
    try:
        query = '''
        select id, name, email from user
        where email = :email
        '''

        result = await conn.execute(
            text(query),
            {
                "email": email
            }
        )

        row = result.fetchone()
        if row is None:
            return None
        
        user = user_schemas.UserDataNotIncludePassword(
            id=row.id,
            name=row.name,
            email=row.email
        )
        result.close()
        
        return user

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    

async def get_user_by_pass(
    conn: Connection,
    email: EmailStr
):
    try:
        query = '''
        select id, name, email, hashed_password from user
        where email = :email
        '''

        result = await conn.execute(
            text(query),
            {
                "email": email
            }
        )

        row = result.fetchone()
        if row is None:
            return None
        
        user = user_schemas.UserDataIncludePassword(
            id=row.id,
            name=row.name,
            email=row.email,
            hashed_password=row.hashed_password
        )
        result.close()
        
        return user

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    


async def register_user(
    conn: Connection,
    name: str,
    email: EmailStr,
    hashed_password: str
):
    try: 
        query = '''
        insert into user(name, email, hashed_password)
        values (:name, :email, :hashed_password)
        '''

        await conn.execute(
            text(query),
            {
                "name": name,
                "email": email,
                "hashed_password": hashed_password
            }
        )
        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    

# 세션 작업
def get_session(request: Request):
    return request.session

def get_session_user_opt(request: Request):
    if "session_user" in request.session.keys():
        return request.session["session_user"]
    return None

def get_session_user_prt(request: Request):
    if "session_user" not in request.session.keys():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="해당 서비스는 로그인이 필요합니다."
        )
    return request.session["session_user"]

def check_valid_auth(session_user: dict, blog_author_id: int, blog_email: str):
    # 범용적으로 사용 가능성 때문
    if session_user is None:
        return False
    
    if session_user["id"] == blog_author_id and session_user["email"] == blog_email:
        return True
    
    return False