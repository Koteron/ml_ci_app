import os
import time
import json
import logging
import threading
import pika
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
TASK_QUEUE = "generation_tasks"
RESULT_QUEUE = "generation_results"

MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LORA_PATH = "./lora-ru-adapter.gguf"

SYSTEM_PROMPT = "Ты — русскоязычный энциклопедический ассистент. Отвечай кратко и точно."


class TextGenerator:
    def __init__(self, model_path: str, lora_path: str | None = None):

        self.llm = Llama(
            model_path=model_path,
            lora_path=lora_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

    def generate(self, prompt: str, params: dict) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=min(int(params.get("max_new_tokens", 200)), 512),
            temperature=max(float(params.get("temperature", 0.3)), 0.1),
            top_p=float(params.get("top_p", 0.85)),
            top_k=int(params.get("top_k", 40)),
            stop=["<|im_end|>"],
        )

        return response["choices"][0]["message"]["content"].strip()


def start_worker():
    generator = TextGenerator(MODEL_PATH, lora_path=LORA_PATH)

    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 120
    params.blocked_connection_timeout = 600

    while True:
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.queue_declare(queue=TASK_QUEUE, durable=True)
            channel.queue_declare(queue=RESULT_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)

            def on_message(ch, method, properties, body):
                data = json.loads(body)
                task_id = data.get("task_id")
                prompt = data.get("prompt", "")
                gen_params = data.get("params", {})

                logging.info(f"Получена задача {task_id}. Длина промпта: {len(prompt)} симв.")

                def work():
                    try:
                        generated_text = generator.generate(prompt, gen_params)
                        response = {
                            "task_id": task_id,
                            "status": "COMPLETED",
                            "result": generated_text,
                            "error": None,
                        }
                    except Exception as e:
                        logging.error(f"Ошибка при генерации для задачи {task_id}: {e}")
                        response = {
                            "task_id": task_id,
                            "status": "FAILED",
                            "result": None,
                            "error": str(e),
                        }

                    def publish_and_ack():
                        try:
                            ch.basic_publish(
                                exchange="",
                                routing_key=RESULT_QUEUE,
                                body=json.dumps(response, ensure_ascii=False),
                                properties=pika.BasicProperties(delivery_mode=2),
                            )
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            logging.info(
                                f"Задача {task_id} завершена со статусом {response['status']}."
                            )
                        except Exception as e:
                            logging.error(
                                f"Не удалось опубликовать/подтвердить задачу {task_id}: {e}"
                            )

                    try:
                        connection.add_callback_threadsafe(publish_and_ack)
                    except Exception as e:
                        logging.error(
                            f"Соединение уже закрыто, результат для {task_id} не отправлен: {e}"
                        )

                threading.Thread(target=work, daemon=True).start()

            channel.basic_consume(queue=TASK_QUEUE, on_message_callback=on_message)
            logging.info("Ожидание задач в очереди RabbitMQ...")
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError:
            logging.warning("Нет соединения с RabbitMQ. Повтор через 5 секунд...")
            time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Остановка по сигналу.")
            break


if __name__ == "__main__":
    start_worker()