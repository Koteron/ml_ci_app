import re
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./model_weights"
TARGET_ARTICLES = 5000
MAX_LENGTH = 512


def clean_wiki_text(text: str) -> str:
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_first_paragraphs(text: str, max_chars: int = 800) -> str:
    sentences = text.split(". ")
    result = ""
    for s in sentences:
        if len(result) + len(s) < max_chars:
            result += s + ". "
        else:
            break
    return result.strip()


def main():
    print("1. Загрузка и фильтрация подмножества датасета...")
    raw_stream = load_dataset("danasone/wikipedia_ru", split="train", streaming=True)
    shuffled_stream = raw_stream.shuffle(seed=42, buffer_size=50000)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_samples = []
    for item in shuffled_stream:
        title = item.get("title", "").strip()
        text = clean_wiki_text(item.get("text", ""))

        summary = get_first_paragraphs(text)
        if len(summary) < 150:
            continue

        messages = [
            {"role": "system", "content": "Ты — русскоязычный энциклопедический ассистент. Отвечай кратко и точно."},
            {"role": "user", "content": f"Что такое {title}?"},
            {"role": "assistant", "content": summary},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_samples.append(chat_text)

        if len(formatted_samples) >= TARGET_ARTICLES:
            break

    print(f"Отобрано {len(formatted_samples)} качественных статей.")
    dataset = Dataset.from_dict({"text": formatted_samples})

    print("2. Токенизация...")
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    print("3. Загрузка модели и подключение LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        save_steps=200,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        train_dataset=tokenized_dataset,
    )

    print("4. Запуск дообучения LoRA...")
    trainer.train()

    print(f"5. Объединение весов и сохранение в {OUTPUT_DIR}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Готово!")


if __name__ == "__main__":
    main()
