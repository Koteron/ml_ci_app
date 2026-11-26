import os
import re
import time
import json
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from llama_cpp import Llama

MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LORA_PATH = "./lora-ru-adapter.gguf"
VAL_FILE = "./val_dataset.jsonl"
SYSTEM_PROMPT = "Ты — русскоязычный энциклопедический ассистент. Отвечай кратко и точно."

SAMPLE_DATA = [
    {
        "prompt": "Что такое фотосинтез?",
        "reference": "Фотосинтез — это процесс образования органических веществ из углекислого газа и воды на свету при участии фотосинтетических пигментов (хлорофилла у растений)."
    },
    {
        "prompt": "Кто написал роман «Преступление и наказание»?",
        "reference": "Роман «Преступление и наказание» написал русский писатель Фёдор Михайлович Достоевский в 1866 году."
    },
    {
        "prompt": "Какая самая длинная река в Европе?",
        "reference": "Самая длинная река в Европе — Волга, её длина составляет 3530 километров."
    },
]


class CyrillicTokenizer:
    def tokenize(self, text: str):
        return re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]+", text.lower())


def load_dataset(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"[!] {path} не найден. Используется встроенный тестовый набор ({len(SAMPLE_DATA)} примеров).\n")
        return SAMPLE_DATA
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_cjk_leak(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def run_inference(llm: Llama, dataset: list[dict], max_tokens: int = 200) -> dict:
    preds = []
    latencies = []
    tps_list = []
    cjk_leaks = 0
    max_token_hits = 0

    for item in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"].strip()},
        ]

        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            stop=["<|im_end|>"],
        )
        elapsed = time.perf_counter() - t0

        text = out["choices"][0]["message"]["content"].strip()
        comp_tokens = out["usage"]["completion_tokens"]

        preds.append(text)
        latencies.append(elapsed)
        tps_list.append(comp_tokens / elapsed if elapsed > 0 else 0)

        if has_cjk_leak(text):
            cjk_leaks += 1
        if comp_tokens >= max_tokens:
            max_token_hits += 1

    return {
        "preds": preds,
        "avg_latency": sum(latencies) / len(latencies),
        "avg_tps": sum(tps_list) / len(tps_list),
        "cjk_leak_rate": (cjk_leaks / len(dataset)) * 100,
        "max_token_rate": (max_token_hits / len(dataset)) * 100,
    }


def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=CyrillicTokenizer())
    rouge_l_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    _, _, f1 = bert_score(preds, refs, lang="ru", verbose=False)
    avg_bert_f1 = f1.mean().item()

    return {
        "rougeL": avg_rouge_l,
        "bleu": bleu,
        "bert_f1": avg_bert_f1,
    }


def main():
    dataset = load_dataset(VAL_FILE)
    refs = [d["reference"] for d in dataset]
    n_threads = int(os.getenv("LLM_THREADS", os.cpu_count() or 4))

    print("1/2 Запуск базовой модели (без LoRA)...")
    base_llm = Llama(model_path=MODEL_PATH, lora_path=None, n_ctx=2048, n_threads=n_threads, verbose=False)
    base_res = run_inference(base_llm, dataset)
    del base_llm

    print("2/2 Запуск дообученной модели (с LoRA)...")
    lora_llm = Llama(model_path=MODEL_PATH, lora_path=LORA_PATH, n_ctx=2048, n_threads=n_threads, verbose=False)
    lora_res = run_inference(lora_llm, dataset)
    del lora_llm

    print("Расчёт метрик (ROUGE-L, BLEU, BERTScore)...\n")
    base_m = compute_metrics(base_res["preds"], refs)
    lora_m = compute_metrics(lora_res["preds"], refs)

    print("=" * 66)
    print(f"{'Метрика':<30} | {'Base Model':<14} | {'LoRA Model':<14}")
    print("-" * 66)
    print(f"{'ROUGE-L (выше = лучше)':<30} | {base_m['rougeL']:<14.4f} | {lora_m['rougeL']:<14.4f}")
    print(f"{'SacreBLEU (выше = лучше)':<30} | {base_m['bleu']:<14.2f} | {lora_m['bleu']:<14.2f}")
    print(f"{'BERTScore F1 (выше = лучше)':<30} | {base_m['bert_f1']:<14.4f} | {lora_m['bert_f1']:<14.4f}")
    print(f"{'Утечка иероглифов (ниже = лучше)':<30} | {base_res['cjk_leak_rate']:<13.1f}% | {lora_res['cjk_leak_rate']:<13.1f}%")
    print(f"{'Упор в max_tokens (ниже = лучше)':<30} | {base_res['max_token_rate']:<13.1f}% | {lora_res['max_token_rate']:<13.1f}%")
    print(f"{'Скорость (токен/сек)':<30} | {base_res['avg_tps']:<14.1f} | {lora_res['avg_tps']:<14.1f}")
    print(f"{'Средняя задержка (сек)':<30} | {base_res['avg_latency']:<14.2f} | {lora_res['avg_latency']:<14.2f}")
    print("=" * 66)

    print("\n--- Примеры ответов (первые 3) ---")
    for i in range(min(3, len(dataset))):
        print(f"\n[Промпт]: {dataset[i]['prompt']}")
        print(f"[Эталон]: {refs[i]}")
        print(f"[Base]  : {base_res['preds'][i]}")
        print(f"[LoRA]  : {lora_res['preds'][i]}")


if __name__ == "__main__":
    main()
