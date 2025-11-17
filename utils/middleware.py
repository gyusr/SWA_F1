from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

# POST => PUT | PATCH | DELETE
class MethodOverrideMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            query = request.query_params
            if query:
                method_override = query["_method"]
                if method_override:
                    method_override = method_override.upper()
                    if method_override in ("PUT", "DELETE"):
                        request.scope["method"] = method_override

        response = await call_next(request)
        return response
