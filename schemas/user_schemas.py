from pydantic import BaseModel, Field, EmailStr

# 검증용
class CorrectUserForm(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    email: EmailStr | None = Field(None)
    password: str | None = Field(None, min_length=2, max_length=30)


# dataclass용
class UserDataNotIncludePassword(BaseModel):
    id: int
    name: str
    email: EmailStr

class UserDataIncludePassword(UserDataNotIncludePassword):
    hashed_password: str
