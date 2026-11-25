import os
import json
import time
import logging
import threading
import pika
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
TASK_QUEUE = "generation_tasks"
RESULT_QUEUE = "generation_results"

LOCAL_MODEL_PATH = "./model_weights"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = "Ты — русскоязычный энциклопедический ассистент. Отвечай кратко и точно."


class TextGenerator:
    def __init__(self):
        model_path = LOCAL_MODEL_PATH if os.path.exists(f"{LOCAL_MODEL_PATH}/config.json") else FALLBACK_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logging.info(f"Загрузка модели из '{model_path}' на устройство: {self.device} ({dtype})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        logging.info("Модель успешно загружена и готова к работе.")

    def generate(self, prompt: str, params: dict) -> str:
        max_new_tokens = min(int(params.get("max_new_tokens", 200)), 512)
        temperature = float(params.get("temperature", 0.3))
        top_p = float(params.get("top_p", 0.85))
        top_k = int(params.get("top_k", 40))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.05,
                temperature=max(temperature, 0.1),
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output_ids[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer


def start_worker():
    generator = TextGenerator()

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