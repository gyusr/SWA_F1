from fastapi import APIRouter, Request, Form, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.exceptions import HTTPException
from pydantic import EmailStr
from sqlalchemy import Connection

from main import templates
from db.database import context_get_conn
from services import user_svc

from passlib.context import CryptContext

router = APIRouter(prefix="/user", tags=["user"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 회원가입
@router.get("/register")
async def register_user_ui(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="register_user.html",
        context={
            
        }
    )    

def get_hashed_password(password: str):
    hashed_password = pwd_context.hash(password)
    return hashed_password

@router.post("/register")
async def register_user(
    name: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    conn: Connection = Depends(context_get_conn)
): 
    # 맞게 입력하였는지 체크
    user = user_svc.correct_user_form(
        name=name,
        email=email,
        password=password
    )

    # 기존에 등록되어 있는 유저인지 확인 (이메일은 중복 안됨)
    user = await user_svc.get_user_by_email(
        conn=conn,
        email=user.email
    )

    if user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="해당 이메일은 이미 등록되어 있습니다."
        )
    
    else: 
        hashed_password = get_hashed_password(password=password)

        await user_svc.register_user(
            conn=conn,
            name=name,
            email=email,
            hashed_password=hashed_password
        )

    return RedirectResponse(
        url="/user/login",
        status_code=status.HTTP_303_SEE_OTHER
    )


# 로그인
@router.get("/login")
async def login_user_ui(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="login_user.html",
        context={

        }
    )

def verify_password(
    plain_password: str,
    hashed_password: str
):
    return pwd_context.verify(plain_password, hashed_password) # True | False

@router.post("/login")
async def login_user(
    request: Request,
    email: EmailStr = Form(...),
    password: str = Form(...),
    conn: Connection = Depends(context_get_conn)
):
    user = user_svc.correct_user_form(
        email=email,
        password=password
    )

    # 이메일 기반 유저 확인
    userpass = await user_svc.get_user_by_pass(
        conn=conn,
        email=email
    )

    if userpass is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="등록되지 않은 사용자입니다."
        )
    
    # 비밀번호 확인
    is_correct_pw = verify_password(
        plain_password=password, 
        hashed_password=userpass.hashed_password
    )

    if not is_correct_pw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비밀번호를 확인해주세요."
        )
    
    # 로그인 통과 / 쿠기-세션 발행
    session = request.session["session_user"] = {
         "id": userpass.id,
         "name": userpass.name,
         "email": userpass.email
    }

    return RedirectResponse(
        url="/meetings/read/all",
        status_code=status.HTTP_303_SEE_OTHER
    )


# 로그아웃
@router.get("/logout")
async def logout(
    request: Request
):
    request.session.clear()
    return RedirectResponse(
        url="/user/login",
        status_code=status.HTTP_303_SEE_OTHER
    )
