# syncscribe-ml

ML-часть SyncScribe. Модуль определяет, затронут ли фрагмент документа изменением источника, и предлагает черновик правки.

## Запуск (Windows, локально)

1. Установи [Ollama](https://ollama.com/download) и скачай модель:
   ```
   ollama pull qwen3:8b
   ```
2. Подготовь окружение (Python 3.10+). Команды выполняются из папки репозитория:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```
3. Сначала прогони 3 примера, чтобы убедиться, что всё работает, потом весь набор:
   ```
   python scripts/run_eval.py --model qwen3:8b --limit 3
   python scripts/run_eval.py --model qwen3:8b
   ```

## Что где

| Путь | Что это |
|---|---|
| `data/eval/gold_v0.jsonl` | Eval-набор: 20 синтетических примеров, описание в `data/eval/README.md` |
| `prompts/impact_v1.md` | Системный промпт. Новая версия промпта кладётся новым файлом (`impact_v2.md`), старый не трогаем |
| `src/syncscribe_ml/schema.py` | Схема ответа модели (JSON Schema для Ollama/vLLM и валидация через pydantic) |
| `src/syncscribe_ml/llm.py` | Один вызов модели на пару (изменение источника, фрагмент) |
| `src/syncscribe_ml/metrics.py` | Accuracy, precision, recall, traceability, латентность |
| `scripts/run_eval.py` | Прогон модели на наборе, результаты сохраняются в `results/` |

## Пороги для пилота

Accuracy ≥ 80%, Precision ≥ 75%, Recall ≥ 75%, Hallucination ≤ 10%, Edit Usefulness ≥ 3.5, Traceability ≥ 60%.
