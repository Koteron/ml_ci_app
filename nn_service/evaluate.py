"""
Оценка качества генерации для проекта Wiki TextGen (nn_service).

Скрипт загружает дообученную модель (или базовую Qwen2.5-1.5B-Instruct),
генерирует ответы на отложенной выборке Wikipedia RU и вычисляет:
  - ROUGE (1/2/L)
  - BLEU
  - BERTScore (F1, с поддержкой русского)
  - Perplexity (на эталонных текстах, для causal LM)

Результаты сохраняются в JSON, примеры генерации — отдельно.

Пример запуска:
    python evaluate_metrics.py \
        --model_path ./model_weights \
        --num_samples 200 \
        --offset 5000 \
        --output_dir ./eval_results
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./model_weights"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "danasone/wikipedia_ru"
SYSTEM_PROMPT = (
    "Ты — русскоязычный энциклопедический ассистент. "
    "Отвечай кратко и точно."
)
MAX_NEW_TOKENS = 200
MAX_INPUT_LENGTH = 512


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


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    path = model_path if os.path.exists(f"{model_path}/config.json") else FALLBACK_MODEL
    logger.info("Загрузка модели из '%s' на %s (%s)...", path, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    logger.info("Модель загружена.")
    return model, tokenizer, device


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    device,
    question: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(device)
    input_len = inputs.input_ids.shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    return answer


def build_eval_set(num_samples: int, offset: int):
    logger.info(
        "Загрузка '%s' (offset=%d, samples=%d)...",
        DATASET_NAME, offset, num_samples,
    )
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    questions, references = [], []
    skipped = 0
    for item in ds:
        if skipped < offset:
            skipped += 1
            continue

        title = item.get("title", "").strip()
        text = clean_wiki_text(item.get("text", ""))
        summary = get_first_paragraphs(text)

        if len(summary) < 150:
            continue

        questions.append(f"Что такое {title}?")
        references.append(summary)

        if len(questions) >= num_samples:
            break

    logger.info("Собрано %d пар (вопрос → эталон).", len(questions))
    return questions, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Путь к дообученной модели (или Hub ID).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Сколько примеров использовать для оценки.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=5000,
        help="Сколько статей пропустить (чтобы не пересекаться с train).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Куда сохранять метрики и примеры.",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="bert-base-multilingual-cased",
        help=(
            "Модель для BERTScore. Для лучшей корреляции с человеком "
            "можно попробовать 'microsoft/deberta-xlarge-mnli' (тяжелее)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Батч для перплексии и BERTScore.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model(args.model_path)

    questions, references = build_eval_set(args.num_samples, args.offset)
    if not references:
        raise RuntimeError("Не удалось собрать ни одного валидного примера.")

    logger.info("Генерация ответов...")
    predictions = []
    for q in tqdm(questions, desc="Generate"):
        try:
            ans = generate_answer(model, tokenizer, device, q)
        except Exception as e:
            logger.warning("Ошибка генерации для '%s': %s", q, e)
            ans = ""
        predictions.append(ans)

    valid = [(p, r) for p, r in zip(predictions, references) if p.strip()]
    if not valid:
        raise RuntimeError("Все сгенерированные ответы пусты — метрики не посчитать.")
    preds_valid, refs_valid = zip(*valid)
    logger.info(
        "Валидных пар: %d из %d (пустых: %d).",
        len(preds_valid), len(predictions), len(predictions) - len(preds_valid),
    )

    results = {"num_samples": len(predictions), "num_valid": len(preds_valid)}

    logger.info("Считаем ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=list(preds_valid),
        references=list(refs_valid),
    )
    results.update({f"rouge_{k}": float(v) for k, v in rouge_res.items()})

    logger.info("Считаем BLEU...")
    bleu = evaluate.load("bleu")
    bleu_res = bleu.compute(
        predictions=list(preds_valid),
        references=[[r] for r in refs_valid],
    )
    results["bleu"] = float(bleu_res["bleu"])

    logger.info("Считаем BERTScore (lang=ru, model=%s)...", args.bertscore_model)
    bertscore = evaluate.load("bertscore")
    bs_res = bertscore.compute(
        predictions=list(preds_valid),
        references=list(refs_valid),
        lang="ru",
        model_type=args.bertscore_model,
        device=device,
        batch_size=args.batch_size,
    )
    results["bertscore_precision"] = float(
        sum(bs_res["precision"]) / len(bs_res["precision"])
    )
    results["bertscore_recall"] = float(
        sum(bs_res["recall"]) / len(bs_res["recall"])
    )
    results["bertscore_f1"] = float(sum(bs_res["f1"]) / len(bs_res["f1"]))

    logger.info("Считаем Perplexity на эталонных текстах...")
    try:
        ppl = evaluate.load("perplexity", module_type="metric")
        ppl_res = ppl.compute(
            model_id=args.model_path,
            predictions=list(refs_valid),
            batch_size=args.batch_size,
            device=device,
        )
        results["perplexity_mean"] = float(ppl_res["mean_perplexity"])
    except Exception as e:
        logger.warning("Не удалось посчитать Perplexity: %s", e)
        results["perplexity_mean"] = None

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Метрики сохранены в %s", metrics_path)

    examples_path = output_dir / "examples.json"
    examples = [
        {
            "question": q,
            "reference": r,
            "prediction": p,
        }
        for q, r, p in zip(questions, references, predictions)
    ]
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logger.info("Примеры сохранены в %s", examples_path)

    print("\n=== Итоговые метрики ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.4f}")
        else:
            print(f"{k:>20}: {v}")


if __name__ == "__main__":
    main()