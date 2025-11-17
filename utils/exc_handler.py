from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from main import templates

# HTTPException Custom Handler
async def custom_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
):
    return templates.TemplateResponse(
        request=request,
        name="http_exception_handler.html",
        context={
            "status_code": exc.status_code,
            "title_message": "불편을 드려 죄송합니다.",
            "detail": exc.detail
        },
        status_code=exc.status_code
    )


# RequestValidationError Custom Handler
async def custom_request_validation_error_handler(
    request: Request,
    exc: RequestValidationError
):
    return templates.TemplateResponse(
        request=request,
        name="request_validation_error_handler.html",
        context={
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "title_message": "잘못된 값을 입력하였습니다.",
            "detail": exc.errors()
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )
