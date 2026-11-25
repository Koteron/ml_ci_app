import asyncio
from contextlib import asynccontextmanager
from uuid import UUID
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .database import engine, Base, get_db
from .models import User, Chat, Message
from .schemas import (
    UserCreate, TokenResponse, ChatCreate, ChatOut, MessageCreate, MessageOut
)
from .auth import hash_password, verify_password, create_access_token, get_current_user
from .rabbitmq import publish_generation_task, consume_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    consumer_task = asyncio.create_task(consume_results())
    yield
    consumer_task.cancel()


app = FastAPI(title="Wiki TextGen User & Chat Service", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:4200",
    "http://45.128.96.119:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    exists = await db.execute(select(User).where(User.email == data.email))
    if exists.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

    user = User(email=data.email, password_hash=hash_password(data.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == form.username))
    user = res.scalar_one_or_none()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    return TokenResponse(access_token=create_access_token(user.id))


@app.get("/chats", response_model=list[ChatOut])
async def get_user_chats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Chat)
        .where(Chat.user_id == current_user.id)
        .order_by(Chat.created_at.desc())
    )
    return res.scalars().all()


@app.post("/chats", response_model=ChatOut, status_code=201)
async def create_chat(
    data: ChatCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = Chat(user_id=current_user.id, title=data.title or "Новый чат")
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


@app.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    await db.delete(chat)
    await db.commit()


@app.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
async def get_chat_messages(
    chat_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Chat)
        .options(selectinload(Chat.messages))
        .where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return chat.messages


@app.post("/chats/{chat_id}/messages", response_model=MessageOut, status_code=202)
async def send_message(
    chat_id: UUID,
    payload: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Chat)
        .options(selectinload(Chat.messages))
        .where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")

    if len(chat.messages) == 0:
        chat.title = payload.content[:45] + ("..." if len(payload.content) > 45 else "")

    user_msg = Message(
        chat_id=chat.id,
        role="user",
        content=payload.content,
        status="COMPLETED",
    )
    db.add(user_msg)

    recent_history = [m.content for m in chat.messages[-3:] if m.status == "COMPLETED"]
    recent_history.append(payload.content)
    compiled_prompt = "\n".join(recent_history)

    params_dict = payload.params.model_dump()
    assistant_msg = Message(
        chat_id=chat.id,
        role="assistant",
        content="",
        status="PENDING",
        params={**params_dict, "_compiled_prompt": compiled_prompt},
    )
    db.add(assistant_msg)
    await db.commit()
    await db.refresh(assistant_msg)

    await publish_generation_task(
        task_id=assistant_msg.id,
        prompt=compiled_prompt,
        params=params_dict,
    )

    return assistant_msg
