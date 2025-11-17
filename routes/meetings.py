from fastapi import APIRouter, Request, status, Depends, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import Connection
from db.database import context_get_conn

from services import meetings_svc, user_svc, processes_svc

router = APIRouter(prefix="/meetings", tags=["meetings"])
templates = Jinja2Templates(directory="templates")


# 회의록 생성
@router.get("/create")
async def create_meeting_ui(
    request: Request,
    session_user = Depends(user_svc.get_session_user_prt)
):
    return templates.TemplateResponse(
        request=request,
        name="create_meeting.html",
        context={
            "session_user": session_user
        }
    )

@router.post("/create")
async def create_meeting(
    request: Request,
    title: str = Form(...), # 조건
    original_meeting: str = Form(...), # 조건
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # 제일 중요한 로직!!!!!
    await meetings_svc.create_meeting(
        conn=conn,
        title=title,
        original_meeting=original_meeting,
        session_user=session_user
    )
    
    return RedirectResponse(
        url="/meetings/read/all",
        status_code=status.HTTP_303_SEE_OTHER
    )


# 회의록 읽기
@router.get("/read/all")
async def get_all_meetings_ui(
    request: Request,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # 모든 회의록 가져오기 (id / title / created_dt) - 본인 것만 가져와야함 / 세션 기반으로
    all_meetings = await meetings_svc.get_all_meetings(
        conn=conn,
        session_user=session_user
    )

    return templates.TemplateResponse(
        request=request,
        name="main_meeting.html",
        context={
            "all_meetings": all_meetings,
            "session_user": session_user
        }
    )


@router.get("/read/{id}")
async def get_by_id_meeting_ui(
    request: Request,
    id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # id 기반 회의록 가져오기 / 세션 기반으로
    meeting = await meetings_svc.get_by_id_meeting(
        conn=conn,
        id=id,
        session_user=session_user
    )
    
    return templates.TemplateResponse(
        request=request,
        name="read_meeting.html",
        context={
            "meeting": meeting,
            "session_user": session_user
        }
    )


# 회의록 수정 - 필요 없을 듯??


# 회의록 삭제
@router.delete("/delete/{id}")
async def delete_meeting(
    request: Request,
    id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
): 
    # 회의록 삭제 로직
    await meetings_svc.delete_meeting(
        conn=conn,
        id=id,
        session_user=session_user
    )

    return RedirectResponse(
        url="/meetings/read/all",
        status_code=status.HTTP_303_SEE_OTHER
    )
