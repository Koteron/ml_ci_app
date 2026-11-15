from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatCreate(BaseModel):
    title: str | None = "Новый чат"

class ChatOut(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    class Config:
        from_attributes = True

class GenerationParams(BaseModel):
    max_new_tokens: int = Field(default=120, ge=20, le=512)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    params: GenerationParams = GenerationParams()

class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    status: str
    created_at: datetime
    class Config:
        from_attributes = True
