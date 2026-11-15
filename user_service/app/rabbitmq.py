import os
import json
import asyncio
import logging
from uuid import UUID
import aio_pika
from sqlalchemy import select
from .database import AsyncSessionLocal
from .models import Message

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
TASK_QUEUE = "generation_tasks"
RESULT_QUEUE = "generation_results"


async def publish_generation_task(task_id: UUID, prompt: str, params: dict):
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.declare_queue(TASK_QUEUE, durable=True)

        payload = {
            "task_id": str(task_id),
            "prompt": prompt,
            "params": params,
        }
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=TASK_QUEUE,
        )


async def consume_results():
    """Фоновая задача: слушает ответы от NN-сервиса и обновляет сообщения в БД."""
    while True:
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            async with connection:
                channel = await connection.channel()
                queue = await channel.declare_queue(RESULT_QUEUE, durable=True)
                logging.info("User Service подключен к очереди generation_results.")

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            data = json.loads(message.body.decode("utf-8"))
                            task_id = UUID(data["task_id"])
                            status = data["status"]
                            raw_result = data.get("result") or ""

                            async with AsyncSessionLocal() as db:
                                res = await db.execute(select(Message).where(Message.id == task_id))
                                msg_record = res.scalar_one_or_none()
                                if msg_record:
                                    msg_record.status = status
                                    if status == "COMPLETED":
                                        # Отрезаем отправленный ранее промпт, оставляя только продолжение
                                        sent_prompt = (msg_record.params or {}).get("_compiled_prompt", "")
                                        clean_reply = raw_result[len(sent_prompt):].strip() if raw_result.startswith(sent_prompt) else raw_result
                                        msg_record.content = clean_reply or raw_result
                                    else:
                                        msg_record.content = f"Ошибка генерации: {data.get('error')}"
                                    await db.commit()
                                    logging.info(f"Сообщение {task_id} обновлено ({status}).")
        except Exception as e:
            logging.warning(f"Ошибка консьюмера RabbitMQ: {e}. Переподключение через 5 сек...")
            await asyncio.sleep(5)
